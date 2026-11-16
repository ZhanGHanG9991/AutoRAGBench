"""Start a frozen submission and send questions one by one. Shared by self_check and job B.

For every question: gateway /admin/open -> POST /answer (timeout) -> gateway /admin/close.
The service sees only PORT, LLM_BASE_URL, CORPUS_DIR, SCRATCH_DIR, HOME, TMPDIR, PATH, LANG, HF_HUB_OFFLINE,
TRANSFORMERS_OFFLINE, CUDA_VISIBLE_DEVICES (no API keys, no proxies). Two launch modes:
  direct  (self_check, already inside the agent sandbox): `env -i ... bash submission/start.sh`; with
          --readonly-workspace the workspace is remounted read-only for the service, as in job B
  sandbox (job B): --service-cmd, normally scripts/run_service.sh; with --service-uds the service has
          no network and is reached over a unix socket (scripts/relay.py).
A crashed service is restarted from the same directory at most --max-restarts times; the question it
crashed on scores 0. With --eval-seconds, the whole evaluation (startup included) ends at that wall-clock limit:
questions not answered by then get error "eval_deadline" and score 0.
Writes --out (one prediction per line) and --out.summary.json.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import time

import httpx

from tasks.common import mean_latency  # evaluator.py lives in the repo root (sys.path[0])


def client(base=None, uds=None, timeout=10):
    if uds:
        return httpx.Client(transport=httpx.HTTPTransport(uds=uds), base_url="http://local", timeout=timeout)
    return httpx.Client(base_url=base, timeout=timeout)


def readonly_workspace_cmd(workspace):
    """Prefix that runs a command with `workspace` read-only, the way job B binds it (scripts/run_service.sh).

    A private mount namespace (unprivileged user namespace, as root of it) remounts the workspace read-only,
    re-enters it (the inherited cwd would still point at the writable mount), then maps back to the caller's
    uid/gid so the service runs as the same user. Every other path keeps its permissions (SCRATCH_DIR stays
    writable).
    """
    script = ('mount --bind "$1" "$1" && mount -o remount,bind,ro "$1" && cd "$1" && '
              'shift && exec unshare --user --map-user="$1" --map-group="$2" "${@:3}"')
    return ["unshare", "--user", "--map-root-user", "--mount", "bash", "-c", script, "readonly-workspace",
            os.path.realpath(workspace), str(os.getuid()), str(os.getgid())]


def readonly_workspace_supported():
    try:
        return subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "true"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class Service:
    def __init__(self, a):
        self.a = a
        self.proc = None
        self.logf = open(a.service_log, "a", buffering=1)

    def env(self):
        e = {
            "PORT": str(self.a.port),
            "LLM_BASE_URL": self.a.llm_base_url,
            "CORPUS_DIR": self.a.corpus_dir,
            "SCRATCH_DIR": self.a.scratch_dir,
            "HOME": self.a.scratch_dir,
            "TMPDIR": os.path.join(self.a.scratch_dir, "tmp"),  # same as scripts/run_service.sh
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        # GPU access inside a sandbox: Singularity --nv exposes the driver libraries via LD_LIBRARY_PATH
        for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"):
            if os.environ.get(k) is not None:
                e[k] = os.environ[k]
        return e

    def start(self, limit=None):
        os.makedirs(os.path.join(self.a.scratch_dir, "tmp"), exist_ok=True)
        limit = self.a.startup_timeout if limit is None else min(limit, self.a.startup_timeout)
        self.logf.write(f"\n=== start {time.strftime('%F %T')} ===\n")
        if self.a.service_cmd:
            cmd, cwd, env = shlex.split(self.a.service_cmd), None, None
        else:
            cmd, cwd, env = ["bash", "submission/start.sh"], self.a.workspace, self.env()
            if self.a.readonly_workspace:
                cmd = readonly_workspace_cmd(self.a.workspace) + cmd
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=self.logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        t0 = time.time()
        with client(self.url, self.a.service_uds, timeout=5) as c:
            while time.time() - t0 < limit:
                if self.proc.poll() is not None:
                    return False, time.time() - t0
                try:
                    if c.get("/health").status_code == 200:
                        return True, time.time() - t0
                except httpx.HTTPError:
                    pass
                time.sleep(2)
        return False, time.time() - t0

    @property
    def url(self):
        return f"http://127.0.0.1:{self.a.port}"

    def client(self, timeout):
        return client(self.url, self.a.service_uds, timeout=timeout)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace=20):
        """Stop the service and return only once it is gone.

        The launched process is not the service: under --service-cmd it is `bash run_service.sh`, which dies on
        SIGTERM at once while the singularity runtime and scripts/relay.py (same process group) keep the previous
        instance serving until it has shut down. Waiting for the process alone let a restart's health check reach
        the old instance through the old socket, and let eval_system.sh delete SCRATCH_DIR under a live service.
        So wait until the whole process group has exited.
        """
        if self.proc is None:
            return
        pgid = self.proc.pid
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            t0 = time.time()
            while time.time() - t0 < grace and self._group_alive(pgid):
                time.sleep(0.2)
            if not self._group_alive(pgid):
                break
        self.proc = None
        if self.a.service_uds and os.path.exists(self.a.service_uds):
            os.remove(self.a.service_uds)  # a restart may only ever reach the new relay

    def _group_alive(self, pgid):
        self.proc.poll()  # reap the launched process itself, or it stays a zombie member of the group
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response-mode", choices=["answer", "evidence"], default="answer")
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gateway", help="gateway base url, e.g. http://127.0.0.1:18300")
    ap.add_argument("--gateway-uds", help="gateway unix socket (eval without network)")
    ap.add_argument("--admin-token-file", help="read, then deleted; default env GATEWAY_ADMIN_TOKEN")
    ap.add_argument("--llm-base-url", default="")
    ap.add_argument("--service-cmd", help="launch command (sandbox mode); default: direct env -i start.sh")
    ap.add_argument("--service-uds", help="reach the service over this unix socket")
    ap.add_argument("--readonly-workspace", action="store_true",
                    help="direct mode: start the service with the workspace remounted read-only (self_check)")
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--scratch-dir", required=True)
    ap.add_argument("--service-log", required=True)
    ap.add_argument("--port", type=int, default=18400)
    ap.add_argument("--startup-timeout", type=float, default=1200)
    ap.add_argument("--answer-timeout", type=float, default=60)
    ap.add_argument("--max-restarts", type=int, default=3)
    ap.add_argument("--eval-seconds", type=float, default=0,
                    help="wall-clock limit of the whole evaluation, startup included (0 = none)")
    a = ap.parse_args()
    if a.readonly_workspace and not readonly_workspace_supported():
        print("evaluator: cannot remount the workspace read-only here (no user namespaces); "
              "refusing a writable fallback", flush=True)
        raise SystemExit(3)
    deadline = time.time() + a.eval_seconds if a.eval_seconds > 0 else None
    left = (lambda: deadline - time.time()) if deadline else (lambda: float("inf"))

    if a.admin_token_file:
        token = open(a.admin_token_file).read().strip()
        os.remove(a.admin_token_file)
    else:
        token = os.environ.pop("GATEWAY_ADMIN_TOKEN")
    admin = {"X-Admin-Token": token}
    gw = client(a.gateway, a.gateway_uds, timeout=10)
    questions = [json.loads(l) for l in open(a.questions) if l.strip()]
    svc = Service(a)
    summary = {"n": len(questions), "restarts": 0, "delivery_failed": False,
               "eval_seconds": a.eval_seconds or None, "n_eval_deadline": 0, "response_mode": a.response_mode}
    ok, summary["startup_seconds"] = svc.start(left())
    if not ok:
        summary["delivery_failed"] = True
    out = open(a.out, "w")
    dead = not ok
    for q in questions:
        rec = {"qid": q["qid"], "answer": None, "citations": None, "error": None, "over_budget": False}
        if dead or left() <= 0:
            rec["error"] = ("eval_deadline" if left() <= 0 else
                            "delivery_failed" if summary["delivery_failed"] else "service_dead")
            summary["n_eval_deadline"] += rec["error"] == "eval_deadline"
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue
        gw.post("/admin/open", json={"qid": q["qid"]}, headers=admin).raise_for_status()
        t0 = time.time()
        try:
            # the last question before the evaluation deadline gets only the time that is left
            with svc.client(min(a.answer_timeout, left())) as c:
                r = c.post("/retrieve" if a.response_mode == "evidence" else "/answer", json={"question": q["question"], "meta": q.get("meta", {})})
            if r.status_code != 200:
                rec["error"] = f"http_{r.status_code}"
            else:
                body = r.json()
                if a.response_mode == "evidence":
                    from multireader.contract import validate_passages
                    rec["passages"] = validate_passages(body.get("passages"))
                elif not isinstance(body.get("answer"), str):
                    rec["error"] = "bad_format"
                else:
                    rec["answer"], rec["citations"] = body["answer"], body.get("citations")
        except httpx.TimeoutException:
            rec["error"] = "timeout"
        except Exception as e:
            rec["error"] = f"exception:{type(e).__name__}"
        rec["latency"] = time.time() - t0
        stats = gw.post("/admin/close", json={"qid": q["qid"]}, headers=admin).json()
        rec["gateway"] = stats
        rec["over_budget"] = bool(stats.get("over_tokens") or stats.get("over_seconds") or rec["latency"] > a.answer_timeout)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        if not svc.alive() or (a.response_mode == "evidence" and rec["error"] == "timeout"):
            if summary["restarts"] >= a.max_restarts:
                dead = True
                continue
            summary["restarts"] += 1
            svc.stop()
            ok, _ = svc.start(left())
            dead = not ok
    svc.stop()
    out.close()
    preds = [json.loads(l) for l in open(a.out)]
    summary["n_errors"] = sum(1 for p in preds if p["error"])
    summary["n_over_budget"] = sum(1 for p in preds if p["over_budget"])
    lat = sorted(p.get("latency", 0) for p in preds if p.get("latency") is not None)
    summary["latency_p50"] = lat[len(lat) // 2] if lat else None
    summary["latency_p95"] = lat[int(len(lat) * 0.95)] if lat else None
    # speed metric (README "Speed"): mean over all questions, unanswered / over-limit ones count as the limit
    summary["latency_mean"], summary["n_latency_capped"] = mean_latency(preds, a.answer_timeout)
    calls = [p["gateway"]["calls"] for p in preds if p.get("gateway")]
    toks = [p["gateway"]["total_tokens"] for p in preds if p.get("gateway")]
    summary["mean_calls"] = sum(calls) / len(calls) if calls else None
    summary["mean_tokens"] = sum(toks) / len(toks) if toks else None
    json.dump(summary, open(a.out + ".summary.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
