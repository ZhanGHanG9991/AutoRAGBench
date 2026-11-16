import os,runpy,sys
from pathlib import Path
import qwen_tok
qwen_tok.QwenPieceTokenizer.__init__.__defaults__=(str(Path(__file__).resolve().parents[1]/"models/qwen"),)
target=sys.argv.pop(1)
runpy.run_path(target,run_name="__main__")
