"""Static scan of a frozen workspace (login node). Writes records/<run>/workspace_scan.json.

- source files (outside .venv) that mention external LLM endpoints or keys
- model directories: weight size (≈ params for bf16/fp16: bytes / 2), architecture, whether the architecture
  generates text, whether the weights fit one H800 (80 GB; the auxiliary-model rule of the task README)
- files that look like raw dataset QA dumps
- with extra roots (e.g. the run's home with its Hugging Face cache): the models found there, as "build_models"
  (models used during the build that the service does not ship)
Usage: scan_workspace.py <workspace> <out.json> [extra_model_root ...]
"""
import json
import os
import re
import sys

LLM_PAT = re.compile(r"api\.deepseek\.com|DEEPSEEK_API_KEY|api\.openai\.com|OPENAI_API_KEY|api\.anthropic\.com|ANTHROPIC_(?:API_KEY|AUTH_TOKEN)|bigmodel\.cn|GLM_API_KEY")
QA_PAT = re.compile(r"musique|question_decomposition|paragraph_support_idx|answerable", re.I)
CODE_EXT = {".py", ".sh", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".txt", ".md", ".cfg"}
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf")
GENERATIVE = ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel", "ForSeq2SeqLM")
H800_BYTES = 80 * 10**9


def size_of(p):  # a dangling symlink (e.g. a copied HF snapshot) holds no bytes
    return os.path.getsize(p) if os.path.exists(p) else 0


def model_info(dp, fns):
    size = sum(size_of(os.path.join(dp, f)) for f in fns if f.endswith(WEIGHT_EXT))
    try:
        cfg = json.load(open(os.path.join(dp, "config.json")))
    except (OSError, ValueError):
        cfg = {}
    arch = cfg.get("architectures") or []
    return {"weight_bytes": size, "approx_params_bf16": size // 2, "model_type": cfg.get("model_type"),
            "architectures": arch, "generative": any(a.endswith(GENERATIVE) for a in arch),
            "fits_one_h800": size <= H800_BYTES, "name_or_path": cfg.get("_name_or_path")}


def scan_models(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        if "config.json" in fns and any(f.endswith(WEIGHT_EXT) for f in fns):
            out[os.path.relpath(dp, root)] = model_info(dp, fns)
    return out


def main():
    ws, out = sys.argv[1], sys.argv[2]
    llm_hits, qa_hits, models = [], [], {}
    for dp, dns, fns in os.walk(ws):
        rel_dp = os.path.relpath(dp, ws)
        in_venv = rel_dp.split(os.sep)[0] in {".venv", "venv", "env"}
        if "config.json" in fns and any(f.endswith(WEIGHT_EXT) for f in fns):
            models[rel_dp] = model_info(dp, fns)
        if in_venv:
            continue
        for f in fns:
            p = os.path.join(dp, f)
            if os.path.splitext(f)[1] not in CODE_EXT or not os.path.isfile(p) or os.path.getsize(p) > 50_000_000:
                continue
            try:
                txt = open(p, errors="ignore").read()
            except OSError:
                continue
            if LLM_PAT.search(txt):
                llm_hits.append({"file": os.path.relpath(p, ws), "matches": sorted(set(LLM_PAT.findall(txt)))[:5]})
            n = len(QA_PAT.findall(txt))
            if n >= 20:
                qa_hits.append({"file": os.path.relpath(p, ws), "count": n})
    build_models = {}
    for root in sys.argv[3:]:
        if os.path.isdir(root):
            build_models.update({f"{root}/{k}": v for k, v in scan_models(root).items()})
    res = {"external_llm_refs": llm_hits, "qa_like_files": qa_hits, "models": models, "build_models": build_models,
           "max_model_params": max((m["approx_params_bf16"] for m in models.values()), default=0),
           "over_one_h800": [k for k, m in {**models, **build_models}.items() if not m["fits_one_h800"]],
           "generative_in_workspace": [k for k, m in models.items() if m["generative"]]}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"external_llm_refs": len(llm_hits), "qa_like_files": len(qa_hits), "models": len(models),
                      "build_models": len(build_models), "max_model_params": res["max_model_params"],
                      "over_one_h800": len(res["over_one_h800"]), "generative_in_workspace": len(res["generative_in_workspace"])}))


if __name__ == "__main__":
    main()
