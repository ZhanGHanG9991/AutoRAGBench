"""Step 1 of the text adaptation of LEGO-GraphRAG: turn a AutoRAGBench corpus into a knowledge graph.

LEGO-GraphRAG assumes an existing KG (Freebase). Our tasks only have text, so the KG is extracted from the
corpus with the fixed generator (Qwen3.5-4B, local vLLM, thinking off, temperature 0) -- the same model and the
same rule as for the agents (indexing-time LLM use only via the fixed generator).

Passages are split exactly like baselines/ragcore.py (200-word windows). One OpenIE call per passage returns
named entities and [subject, relation, object] triples (HippoRAG-style one-shot prompt). Results are appended to
openie.jsonl, so the job can be resumed.

  python openie.py --corpus <corpus_dir> --out <index_dir> --llm http://127.0.0.1:PORT/v1 --key KEY
"""
import argparse
import asyncio
import json
import os
import re
import time

import httpx

SYSTEM = ("Your task is to construct an RDF (Resource Description Framework) graph from the given passage. "
          "First list the named entities of the passage, then extract triples [subject, relation, object]. "
          "Each triple should contain at least one, and preferably two, of the named entities. "
          "Resolve pronouns to the specific names they refer to. Reply with JSON only.")

EX_PASSAGE = ("Radio City\nRadio City is India's first private FM radio station and was started on 3 July 2001. "
              "It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 "
              "with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, "
              "songs, and other music-related features.")
EX_OUTPUT = json.dumps({
    "named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"],
    "triples": [["Radio City", "located in", "India"], ["Radio City", "is", "private FM radio station"],
                ["Radio City", "started on", "3 July 2001"], ["Radio City", "plays songs in", "Hindi"],
                ["Radio City", "plays songs in", "English"], ["Radio City", "forayed into", "New Media"],
                ["Radio City", "launched", "PlanetRadiocity.com"], ["PlanetRadiocity.com", "launched in", "May 2008"],
                ["PlanetRadiocity.com", "is", "music portal"], ["PlanetRadiocity.com", "offers", "news"],
                ["PlanetRadiocity.com", "offers", "videos"], ["PlanetRadiocity.com", "offers", "songs"]]})


def load_passages(corpus_dir, words=200):  # identical to baselines/ragcore.py
    passages = []
    for line in open(os.path.join(corpus_dir, "corpus.jsonl")):
        d = json.loads(line)
        toks = d["text"].split()
        for s in range(0, max(len(toks), 1), words):
            passages.append((d["doc_id"], d["title"], " ".join(toks[s:s + words])))
    return passages


def parse(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            obj = {}
    ents = [e for e in obj.get("named_entities", []) if isinstance(e, str) and e.strip()]
    triples = [[str(x).strip() for x in t] for t in obj.get("triples", [])
               if isinstance(t, list) and len(t) == 3 and all(str(x).strip() for x in t)]
    return ents, triples


async def extract(client, sem, url, key, pid, title, text, out, stats):
    passage = f"{title}\n{text}" if title else text
    payload = {"model": "generator", "temperature": 0, "max_tokens": 1536,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"Passage:\n{EX_PASSAGE}"},
                            {"role": "assistant", "content": EX_OUTPUT},
                            {"role": "user", "content": f"Passage:\n{passage}"}]}
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(f"{url}/chat/completions", json=payload,
                                      headers={"Authorization": f"Bearer {key}"}, timeout=600)
                r.raise_for_status()
                j = r.json()
                break
            except Exception as ex:  # noqa: BLE001 -- retried, then recorded as a failure
                if attempt == 2:
                    out.write(json.dumps({"pid": pid, "error": str(ex)[:200]}) + "\n")
                    stats["fail"] += 1
                    return
                await asyncio.sleep(5)
    content = j["choices"][0]["message"].get("content") or ""
    ents, triples = parse(content)
    stats["done"] += 1
    stats["out_tok"] += j.get("usage", {}).get("completion_tokens", 0)
    stats["in_tok"] += j.get("usage", {}).get("prompt_tokens", 0)
    stats["empty"] += not triples
    out.write(json.dumps({"pid": pid, "entities": ents, "triples": triples, "raw": None if triples else content[:500]},
                         ensure_ascii=False) + "\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="only the first N passages (throughput probe)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    passages = load_passages(a.corpus)
    with open(os.path.join(a.out, "passages.jsonl"), "w") as f:
        for pid, (doc, title, text) in enumerate(passages):
            f.write(json.dumps({"pid": pid, "doc_id": doc, "title": title, "text": text}, ensure_ascii=False) + "\n")
    path = os.path.join(a.out, "openie.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if "error" not in r:
                done.add(r["pid"])
    todo = [i for i in range(len(passages)) if i not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"passages={len(passages)} done={len(done)} todo={len(todo)}", flush=True)
    stats = {"done": 0, "fail": 0, "empty": 0, "in_tok": 0, "out_tok": 0}
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    limits = httpx.Limits(max_connections=a.concurrency + 8, max_keepalive_connections=a.concurrency + 8)
    async with httpx.AsyncClient(limits=limits) as client:
        with open(path, "a") as out:
            tasks = [asyncio.create_task(extract(client, sem, a.llm, a.key, i, passages[i][1], passages[i][2], out, stats))
                     for i in todo]
            pending = set(tasks)
            while pending:
                _, pending = await asyncio.wait(pending, timeout=60)
                dt = time.time() - t0
                print(f"[{dt:7.0f}s] {stats['done']}/{len(todo)} fail={stats['fail']} empty={stats['empty']} "
                      f"out_tok/s={stats['out_tok'] / dt:.0f} in_tok/s={stats['in_tok'] / dt:.0f}", flush=True)
                out.flush()
    dt = time.time() - t0
    summary = {**stats, "passages": len(passages), "seconds": round(dt), "out_tok_per_s": round(stats["out_tok"] / dt)}
    json.dump(summary, open(os.path.join(a.out, f"openie_summary{'_probe' if a.limit else ''}.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
