"""Session segments and reprompt statistics of one agent run, merged into records/<run>/agent_meta.json.

  agent_segments.py <agent_stderr.log> <agent_trace.jsonl> <agent_meta.json>

The harness adapters (agents/harness/*.sh) log "[ts] <cli> <segment> (left=Ns)" when a segment starts and
"[ts] <cli> <segment> exit=N" when it ends; segment "initial" is the first prompt, "resume-i" a reprompt.
A segment ends voluntarily (exit 0), at the recorded deadline (exit 124 / 137 from timeout) or by a crash (anything else).
Usage-limit / auth / stream errors are matched in the trace and the launcher log (known
provider limit messages). "platform_suspect" marks a run the general judge and a human should look at:
an abnormal final segment, more than 3 crashed segments, the rapid-exit guard stopped the reprompt loop, or a usage-limit / balance error.
"""
import json
import re
import sys
from datetime import datetime

START = re.compile(r"^\[(?P<ts>[^\]]+)\] \S+ (?P<seg>initial|resume-\d+) \(left=(?P<left>\d+)s\)")
END = re.compile(r"^\[(?P<ts>[^\]]+)\] \S+ (?P<seg>initial|resume-\d+) exit=(?P<code>-?\d+)")
LIMIT_PATTERNS = ["You've hit your limit", "You've hit your session limit", "spending_limit", "billing_hard_limit",
                  "insufficient_quota", "Quota exceeded", "budget_exceeded", "token_expired",
                  "Failed to refresh token", "Reconnecting... 5/5", "Insufficient Balance", "rate_limit_exceeded",
                  "authentication_error", "invalid_api_key",
                  # Zhipu GLM via Claude Code: account out of balance ("[1113][余额不足或无可用资源包,请充值。]")
                  "余额不足", "[1113]", "Request rejected (429)", "\"terminal_reason\":\"api_error\""]
RAPID = 60
DEADLINE_CODES = {124, 137}


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def main():
    log, trace, meta_path = sys.argv[1:4]
    segs, open_ = [], {}
    text = open(log, errors="ignore").read()
    for line in text.splitlines():
        m = START.match(line)
        if m:
            open_[m["seg"]] = {"name": m["seg"], "start": m["ts"], "left_at_start": int(m["left"])}
            continue
        m = END.match(line)
        if m and m["seg"] in open_:
            s = open_.pop(m["seg"])
            s.update(end=m["ts"], exit=int(m["code"]), seconds=round(ts(m["ts"]) - ts(s["start"])))
            segs.append(s)
    segs += [{**s, "end": None, "exit": None, "seconds": None} for s in open_.values()]  # killed without an end line
    hits = set()
    for src in (log, trace):
        try:
            with open(src, errors="ignore") as f:
                for line in f:
                    hits.update(p for p in LIMIT_PATTERNS if p in line)
        except OSError:
            pass
    meta = json.load(open(meta_path))
    budget = (ts(meta["deadline"]) - ts(segs[0]["start"])) if segs and meta.get("deadline") else None
    def deadline_stop(segment):
        # SIGKILL/137 before the recorded deadline is not a normal budget timeout.
        # Allow two seconds of rounding between the shell and timestamp logging.
        return (segment['exit'] in DEADLINE_CODES and segment.get('end') and meta.get('deadline')
                and ts(segment['end']) >= ts(meta['deadline']) - 2)
    crashed = [s for s in segs if s['exit'] not in (0, None) and not deadline_stop(s)]
    stats = {
        "segments": segs,
        "n_reprompts": sum(s["name"].startswith("resume-") for s in segs),
        "n_crashed_segments": len(crashed),
        "n_rapid_exits": sum(1 for s in segs if s["seconds"] is not None and s["seconds"] < RAPID),
        "active_fraction": round(sum(s["seconds"] or 0 for s in segs) / budget, 3) if budget else None,
        "first_stop_seconds": next((s["seconds"] for s in segs if s["name"] == "initial" and s["exit"] == 0), None),
        "reprompt_stopped_rapid_exits": "REPROMPT_STOPPED" in text,
        "limit_hits": sorted(hits),
    }
    stats["platform_suspect"] = (stats["n_crashed_segments"] > 3 or stats["reprompt_stopped_rapid_exits"]
                                 or bool(stats["limit_hits"]) or bool(segs and segs[-1] in crashed))
    meta["session_stats"] = stats
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "segments"}))


if __name__ == "__main__":
    main()
