"""Turn an agent trace (Codex `exec --json`, Claude Code `stream-json`, DeepSeek Harness `headless --json` or
opencode `run --format json`) into readable text and a
usage summary.  parse_trace.py <harness> <agent_trace.jsonl> -o solve_parsed.txt [--usage usage.json]"""
import argparse
import json

CLIP = 4000


def clip(s, n=CLIP):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f"\n... [{len(s) - n} more chars]"


def events(path):
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def parse_codex(path):
    out, usage, turns, cmds = [], {}, 0, 0
    for e in events(path):
        t = e.get("type")
        if t == "turn.started":
            turns += 1
            out.append(f"\n===== turn {turns} =====")
        elif t == "turn.completed":
            for k, v in (e.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
        elif t == "item.completed":
            it = e.get("item") or {}
            k = it.get("type")
            if k == "agent_message":
                out.append(f"[agent] {it.get('text', '')}")
            elif k == "reasoning":
                out.append(f"[thinking] {clip(it.get('text', ''), 1500)}")
            elif k == "command_execution":
                cmds += 1
                out.append(f"[cmd] $ {it.get('command', '')}\n[exit {it.get('exit_code')}] {clip(it.get('aggregated_output', ''))}")
            elif k == "file_change":
                out.append("[files] " + ", ".join(f"{c.get('kind')} {c.get('path')}" for c in it.get("changes", [])))
            else:
                out.append(f"[{k}] {clip(it)}")
    return out, {"turns": turns, "commands": cmds, **usage}


def parse_claude(path):
    """Usage: Claude Code reports a session's token totals only in its `result` event (assistant messages of the
    DeepSeek / GLM endpoints carry output 0, GLM's carry nothing, and one API call can appear as several assistant
    events). A segment killed at the deadline has no `result`: its input / cache tokens come from the assistant
    messages (deduplicated by message id), its output is unknown -> counted in segments_without_result."""
    out, usage, turns, cmds, result = [], {}, 0, 0, {}
    seg = {}  # message id -> usage, since the last `result`
    no_result = 0

    def add(u):
        for k, v in (u or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
    for e in events(path):
        t = e.get("type")
        if t == "assistant":
            turns += 1
            msg = e.get("message") or {}
            seg[msg.get("id")] = msg.get("usage") or {}
            for c in msg.get("content", []):
                if c.get("type") == "text":
                    out.append(f"[agent] {c.get('text', '')}")
                elif c.get("type") == "thinking":
                    out.append(f"[thinking] {clip(c.get('thinking', ''), 1500)}")
                elif c.get("type") == "tool_use":
                    cmds += 1
                    out.append(f"[tool {c.get('name')}] {clip(c.get('input'))}")
        elif t == "user":
            for c in (e.get("message") or {}).get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    out.append(f"[result] {clip(c.get('content'))}")
        elif t == "result":
            result = {k: e.get(k) for k in ("subtype", "num_turns", "duration_ms", "total_cost_usd", "is_error")}
            out.append(f"\n===== result: {json.dumps(result)} =====")
            add(e.get("usage"))
            seg = {}
    if seg:  # last segment ended without a result event (killed at the deadline)
        no_result = 1
        for u in seg.values():
            add({k: v for k, v in u.items() if k != "output_tokens"})
    return out, {"turns": turns, "tool_calls": cmds, **usage, "segments_without_result": no_result, "result": result}


def parse_dsh(path):
    out, usage, turns, cmds, ends = [], {}, 0, 0, []
    for e in events(path):
        t = e.get("type")
        if t == "session":
            out.append(f"\n===== session {e.get('sessionId')} =====")
        elif t == "status" and e.get("phase") == "turn_start":
            turns += 1
            out.append(f"\n===== turn {turns} =====")
        elif t == "status" and e.get("phase") == "step_end":
            for k, v in (e.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
        elif t == "status" and e.get("phase") == "turn_end":
            ends.append((e.get("reason") or {}).get("kind"))
        elif t == "text":
            out.append(f"[agent] {e.get('text', '')}")
        elif t == "thinking":
            out.append(f"[thinking] {clip(e.get('text', ''), 1500)}")
        elif t == "tool_call":
            cmds += 1
            out.append(f"[tool {e.get('tool')}] {clip(e.get('input'))}")
        elif t == "tool_result":
            out.append(f"[result {e.get('status')}] {clip(e.get('result'))}")
        elif t == "error":
            out.append(f"[error] {clip(e)}")
    return out, {"turns": turns, "tool_calls": cmds, **usage, "turn_ends": ends}


def parse_opencode(path):
    out, usage, sessions, cmds, cost = [], {}, [], 0, 0.0
    for e in events(path):
        t, p = e.get("type"), e.get("part") or {}
        if e.get("sessionID") and e.get("sessionID") not in sessions:
            sessions.append(e["sessionID"])
            out.append(f"\n===== session {e['sessionID']} =====")
        if t == "step_finish":
            tok = p.get("tokens") or {}
            for k, v in tok.items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
            for k, v in (tok.get("cache") or {}).items():
                usage[f"cache_{k}"] = usage.get(f"cache_{k}", 0) + v
            cost += p.get("cost") or 0
        elif t == "text":
            out.append(f"[agent] {p.get('text', '')}")
        elif t == "reasoning":
            out.append(f"[thinking] {clip(p.get('text', ''), 1500)}")
        elif t == "tool_use":
            cmds += 1
            st = p.get("state") or {}
            out.append(f"[tool {p.get('tool')}] {clip(st.get('input'))}\n[result {st.get('status')}] "
                       f"{clip(st.get('output') if 'output' in st else st.get('error'))}")
        elif t == "error":
            out.append(f"[error] {clip(e.get('error') or e)}")
    return out, {"sessions": len(sessions), "tool_calls": cmds, **usage, "cost_usd": round(cost, 4)}


PARSERS = {"codex": parse_codex, "claude_code": parse_claude, "deepseek_harness": parse_dsh, "opencode": parse_opencode}


def unified(harness, u):
    """The harnesses' usage fields in one schema (all tokens of all model calls of the agent):
    input_tokens = every prompt token (cache hits included), cached_input_tokens = cache hits,
    cache_write_tokens, output_tokens = every generated token (reasoning included)."""
    g = lambda *ks: sum((u.get(k) or 0) for k in ks)
    if harness == "codex":  # OpenAI usage: input includes cached, output includes reasoning
        r = {"input_tokens": g("input_tokens"), "cached_input_tokens": g("cached_input_tokens"),
             "cache_write_tokens": g("cache_write_input_tokens"), "output_tokens": g("output_tokens")}
    elif harness == "claude_code":  # Anthropic usage: input excludes cache reads / writes
        r = {"input_tokens": g("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"),
             "cached_input_tokens": g("cache_read_input_tokens"), "cache_write_tokens": g("cache_creation_input_tokens"),
             "output_tokens": g("output_tokens")}
    elif harness == "deepseek_harness":  # totalTokens = inputTokens + outputTokens + cacheReadTokens
        r = {"input_tokens": g("inputTokens", "cacheReadTokens", "cacheWriteTokens"), "cached_input_tokens": g("cacheReadTokens"),
             "cache_write_tokens": g("cacheWriteTokens"), "output_tokens": g("outputTokens")}
    elif harness == "opencode":  # total = input + output + reasoning + cache.read (+ cache.write)
        r = {"input_tokens": g("input", "cache_read", "cache_write"), "cached_input_tokens": g("cache_read"),
             "cache_write_tokens": g("cache_write"), "output_tokens": g("output", "reasoning")}
    else:
        return None
    r["output_incomplete"] = bool(u.get("segments_without_result"))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("harness", choices=sorted(PARSERS))
    ap.add_argument("trace")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--usage")
    a = ap.parse_args()
    lines, usage = PARSERS[a.harness](a.trace)
    usage["unified"] = unified(a.harness, usage)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    if a.usage:
        json.dump(usage, open(a.usage, "w"), indent=1)
    print(json.dumps(usage))


if __name__ == "__main__":
    main()
