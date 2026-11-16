#!/usr/bin/env python3
"""Workspace/runtime assembly helpers from the formal-v2 packager."""
import datetime as dt
import hashlib,json,os,re,shlex,shutil,subprocess as sp
from pathlib import Path
R = Path(os.environ.get("AUTORAGBENCH_ROOT", ".")).resolve()

def run(args, **kw):
    print("+", shlex.join(map(str, args)), flush=True)
    return sp.run(list(map(str, args)), check=True, **kw)

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(value)

def dump(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_env(ws, pythonpath=""):
    return dict(
        PATH="/usr/local/bin:/usr/bin:/bin",
        HOME=str(ws),
        TMPDIR=str(ws),
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHON_DOTENV_DISABLED="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=pythonpath,
    )

def copy_code(src, dst):
    src, dst = Path(src), Path(dst)
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for name in files:
            if name.endswith(".py"):
                p = Path(root) / name
                q = dst / p.relative_to(src)
                q.parent.mkdir(parents=True, exist_ok=True)
                assert not q.exists(), q
                shutil.copy2(p, q)

def copy_venv(src, dst):
    dst.mkdir()
    for p in sorted(src.iterdir()):
        if p.name == "lib64" and p.is_symlink() and os.readlink(p) == "lib":
            (dst / "lib64").symlink_to("lib")
        else:
            run(["cp", "-aL", p, dst / p.name])

def copy_runtime(srcvenv, ws):
    cfg = (srcvenv / "pyvenv.cfg").read_text()
    home = re.search(r"^home\s*=\s*(.+)$", cfg, re.M).group(1)
    base = Path(home).resolve().parent
    run(["cp", "-aL", base, ws / "python"])
    copy_venv(srcvenv, ws / ".venv")
    stamp = dt.datetime.now().strftime("20261025-%H%M%S")
    pcfg = ws / ".venv/pyvenv.cfg"
    shutil.copy2(pcfg, str(pcfg) + ".bak-" + stamp)
    pcfg.chmod(pcfg.stat().st_mode | 0o200)
    (ws / ".venv/bin").chmod((ws / ".venv/bin").stat().st_mode | 0o200)
    pcfg.write_text(re.sub(r"^home\s*=.*$", "home = " + str(ws / "python/bin"), cfg, flags=re.M))
    for name in ("python", "python3", "python3.11"):
        p = ws / ".venv/bin" / name
        if p.exists() or p.is_symlink():
            p.rename(str(p) + ".bak-" + stamp)
        p.symlink_to(ws / "python/bin/python3.11")
    for p in (ws / ".venv/lib/python3.11/site-packages").glob("__editable__*finder.py"):
        old = p.read_text()
        new = old.replace(str(R / "humansota/LightRAG"), str(ws / "upstream"))
        new = new.replace(str((R / "humansota/LightRAG").resolve()), str(ws / "upstream"))
        if old != new:
            shutil.copy2(p, str(p) + ".bak-" + stamp)
            p.write_text(new)
    return ws / ".venv/bin/python"

def copy_bge_runtime(ws):
    src = R / "envs/rag-base"
    home = re.search(r"^home\s*=\s*(.+)$", (src / "pyvenv.cfg").read_text(), re.M).group(1)
    run(["cp", "-aL", Path(home).resolve().parent, ws / "bge-python"])
    copy_venv(src, ws / "bge-venv")
    cfg = ws / "bge-venv/pyvenv.cfg"
    stamp = dt.datetime.now().strftime("20261025-%H%M%S")
    shutil.copy2(cfg, str(cfg) + ".bak-" + stamp)
    cfg.chmod(cfg.stat().st_mode | 0o200)
    (ws / "bge-venv/bin").chmod((ws / "bge-venv/bin").stat().st_mode | 0o200)
    cfg.write_text(re.sub(r"^home\s*=.*$", "home = " + str(ws / "bge-python/bin"), cfg.read_text(), flags=re.M))
    text = cfg.read_text()
    major = ".".join(re.search(r"version_info\s*=\s*([0-9.]+)", text).group(1).split(".")[:2]) if "version_info" in text else "3.12"
    for n in ("python", "python3", f"python{major}"):
        p = ws / "bge-venv/bin" / n
        if p.exists() or p.is_symlink():
            p.rename(str(p) + ".bak-" + stamp)
        p.symlink_to(ws / f"bge-python/bin/python{major}")
    py = ws / "bge-venv/bin/python"
    run([py, "-c", "import torch,transformers; print(torch.__version__, transformers.__version__)"], env=clean_env(ws))
    return py
