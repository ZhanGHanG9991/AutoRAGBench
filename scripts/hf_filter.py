"""Hugging Face endpoint for agents that serves models but no datasets (run.sbatch, host side).

  hf_filter.py --port P --upstream https://hf-mirror.com --log <file> [--cache DIR]

Agents get HF_ENDPOINT=http://127.0.0.1:P. Requests for datasets (/datasets/..., /api/datasets...) get 403: the
benchmarks' public dev/test files hold the hidden test answers. Everything else is forwarded unchanged (redirects
are passed through, not followed). The agent could still reach the mirror directly; this only removes the
default path (e.g. datasets.load_dataset) — the data_lookup judge covers the rest. Every request is logged.

Robustness (added 2026-10-21, after bursts of failed requests when many runs start at once):
  - idempotent requests (GET/HEAD) are retried on connection errors and on upstream 429/502/503/504 before the
    error is passed on;
  - --cache DIR: HTTP 200 responses for immutable, commit-addressed content (model files under .../resolve/<commit>/...,
    the mirror's /api/resolve-cache/models/... and per-commit API listings) are stored once in DIR, shared by all
    runs, and served from there. Requests for a branch (e.g. "main") and range requests always go upstream.
    Concurrent identical requests wait for the first one instead of all downloading (lock file per entry).
    The cache changes no request's result, only how fast and how reliably it arrives.
"""
import argparse
import hashlib
import http.server
import json
import os
import re
import shutil
import socketserver
import time
import urllib.error
import urllib.request

BLOCK = re.compile(r"^/+(api/+)?datasets(/|$|\?)", re.I)
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length"}
COMMIT = r"[0-9a-f]{40}"
IMMUTABLE = re.compile(rf"^/+(api/+resolve-cache/+models/+[^?]+/{COMMIT}/|api/+models/+[^?]+/+(revision|tree)/+{COMMIT}"
                       rf"|(?!api/)[^/?]+/+[^/?]+/+resolve/+{COMMIT}/)")
RETRY_STATUS = {429, 502, 503, 504}
# A commit fixes file content, not the lifetime of a CDN signed URL. Never
# persist redirects, including HEAD redirects containing Xet download tokens.
CACHE_STATUS = {200}
LOCK_WAIT = 3600       # a follower waits at most this long for the leader of the same entry
LOCK_STALE = 3600      # a lock file older than this without an entry is taken over


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


OPENER = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited bodies: no re-chunking needed

    def log_message(self, *a):
        pass

    def _log(self, status, blocked, **extra):
        with open(self.server.logfile, "a") as f:
            f.write(json.dumps({"ts": time.time(), "method": self.command, "path": self.path, "status": status,
                                "blocked": blocked, **extra}) + "\n")

    # ------------------------------------------------------------------ upstream
    def _open(self, body):
        """Upstream response (HTTPError for 3xx/4xx/5xx) and the number of retries, or raise after the last try."""
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        retry = self.command in ("GET", "HEAD")
        delays = [1, 3, 8] if retry else []
        for attempt in range(len(delays) + 1):
            req = urllib.request.Request(self.server.upstream + self.path, data=body, headers=headers,
                                         method=self.command)
            try:
                resp = OPENER.open(req, timeout=120)
            except urllib.error.HTTPError as e:  # 3xx (not followed), 4xx, 5xx: relay as is (after retries)
                resp = e
            except Exception:  # noqa: BLE001  connection error / timeout
                if attempt < len(delays):
                    time.sleep(delays[attempt])
                    continue
                raise
            code = getattr(resp, "status", None) or resp.code
            if code in RETRY_STATUS and attempt < len(delays):
                wait = delays[attempt]
                try:
                    wait = min(30, max(wait, int(resp.headers.get("Retry-After") or 0)))
                except ValueError:
                    pass
                resp.close()
                time.sleep(wait)
                continue
            return resp, attempt
        raise RuntimeError("unreachable")

    def _headers_of(self, resp):
        out = []
        for k, v in resp.headers.items():
            if k.lower() in HOP:
                continue
            if k.lower() == "location" and v.startswith(self.server.upstream):
                v = v[len(self.server.upstream):] or "/"  # keep same-host redirects on the filter
            out.append([k, v])
        if resp.headers.get("Content-Length"):
            out.append(["Content-Length", resp.headers["Content-Length"]])
        return out

    def _send_head(self, status, headers):
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()

    # ------------------------------------------------------------------ cache
    def _has_cached(self, meta_p):
        """Ignore old redirect entries without trusting their signed URLs."""
        try:
            with open(meta_p) as f:
                return json.load(f).get("status") in CACHE_STATUS
        except (OSError, ValueError, TypeError, AttributeError):
            return False

    def _cacheable(self):
        return (self.server.cache and self.command in ("GET", "HEAD") and "Range" not in self.headers
                and "Authorization" not in self.headers and IMMUTABLE.match(self.path))

    def _paths(self):
        key = hashlib.sha256(f"{self.command} {self.path}".encode()).hexdigest()
        d = os.path.join(self.server.cache, key[:2])
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, key + ".meta"), os.path.join(d, key + ".body"), os.path.join(d, key + ".lock")

    def _serve_cached(self, meta_p, body_p):
        with open(meta_p) as f:
            meta = json.load(f)
        self._send_head(meta["status"], meta["headers"])
        if self.command != "HEAD" and meta.get("body"):
            try:
                with open(body_p, "rb") as f:
                    shutil.copyfileobj(f, self.wfile, 1 << 20)
            except (BrokenPipeError, ConnectionResetError):
                pass
        self._log(meta["status"], False, cache="hit")

    def _take_lock(self, lock_p, meta_p):
        """True: this request is the leader (fetches and stores); False: an entry appeared while waiting."""
        start = time.time()
        while True:
            try:
                fd = os.open(lock_p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time()}".encode())
                os.close(fd)
                if self._has_cached(meta_p):  # stored between our check and the lock
                    os.unlink(lock_p)
                    return False
                return True
            except FileExistsError:
                if self._has_cached(meta_p):
                    return False
                try:
                    if time.time() - os.path.getmtime(lock_p) > LOCK_STALE:
                        os.unlink(lock_p)  # leader died: take over
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - start > LOCK_WAIT:
                    return True  # give up waiting: fetch ourselves (without storing twice: replace is atomic)
                time.sleep(1)

    def _forward_cached(self):
        meta_p, body_p, lock_p = self._paths()
        if self._has_cached(meta_p):
            return self._serve_cached(meta_p, body_p)
        if not self._take_lock(lock_p, meta_p):
            return self._serve_cached(meta_p, body_p)
        try:
            if self._has_cached(meta_p):
                return self._serve_cached(meta_p, body_p)
            try:
                resp, retries = self._open(None)
            except Exception as e:  # noqa: BLE001
                self.send_error(502, f"upstream: {e}")
                return self._log(502, False, cache="miss", error=str(e)[:200])
            code = getattr(resp, "status", None) or resp.code
            headers = self._headers_of(resp)
            self._send_head(code, headers)
            store = code in CACHE_STATUS
            tmp = f"{body_p}.{os.getpid()}.{id(self)}.part"
            n, expected = 0, resp.headers.get("Content-Length")
            out = open(tmp, "wb") if store and self.command == "GET" else None
            client_ok = True
            try:
                if self.command != "HEAD":
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        n += len(chunk)
                        if out:
                            out.write(chunk)
                        if client_ok:
                            try:
                                self.wfile.write(chunk)
                            except (BrokenPipeError, ConnectionResetError):
                                client_ok = False  # keep reading so that the entry is complete for others
                                if not out:
                                    break
            finally:
                if out:
                    out.close()
                resp.close()
            complete = expected is None or self.command == "HEAD" or n == int(expected)
            if store and complete:
                if out:
                    os.replace(tmp, body_p)
                meta = {"status": code, "headers": headers, "body": bool(out), "path": self.path,
                        "stored": time.time(), "bytes": n}
                mtmp = f"{meta_p}.{os.getpid()}.{id(self)}.part"
                with open(mtmp, "w") as f:
                    json.dump(meta, f)
                os.replace(mtmp, meta_p)
            elif os.path.exists(tmp):
                os.unlink(tmp)
            self._log(code, False, cache="miss", stored=bool(store and complete), retries=retries)
        finally:
            try:
                os.unlink(lock_p)
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------ requests
    def _forward(self, body=None):
        if BLOCK.match(self.path):
            msg = b'{"error": "dataset downloads are disabled in this environment"}'
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(msg)
            self._log(403, True)
            return
        if self._cacheable():
            return self._forward_cached()
        try:
            resp, retries = self._open(body)
        except Exception as e:  # noqa: BLE001
            self.send_error(502, f"upstream: {e}")
            self._log(502, False, error=str(e)[:200])
            return
        code = getattr(resp, "status", None) or resp.code
        self._send_head(code, self._headers_of(resp))
        try:
            if self.command != "HEAD":
                shutil.copyfileobj(resp, self.wfile, 1 << 20)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()
        self._log(code, False, retries=retries) if retries else self._log(code, False)

    def do_GET(self):
        self._forward()

    def do_HEAD(self):
        self._forward()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self._forward(self.rfile.read(n) if n else b"")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--cache", help="shared cache directory for immutable (commit-addressed) content")
    a = ap.parse_args()
    srv = Server(("127.0.0.1", a.port), Handler)
    srv.upstream, srv.logfile = a.upstream.rstrip("/"), a.log
    srv.cache = os.path.abspath(a.cache) if a.cache else None
    if srv.cache:
        os.makedirs(srv.cache, exist_ok=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
