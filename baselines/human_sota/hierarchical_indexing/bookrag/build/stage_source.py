"""Stage the patched upstream BookRAG package and its license notices."""
from pathlib import Path
import shutil
b = Path(__file__).resolve().parents[1]
dst = b / "rageng/vendor"
shutil.copytree(b / "Core", dst / "Core", dirs_exist_ok=True)
for name in ["LICENSE", "LICENSE_SCOPE.md", "THIRD_PARTY_NOTICES.md"]:
    shutil.copy2(b / name, dst / name)
shutil.copytree(b / "LICENSES", dst / "LICENSES", dirs_exist_ok=True)
