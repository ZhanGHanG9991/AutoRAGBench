
def _release_path(value):
    """Resolve portable build/workspace paths without changing their layout."""
    import os
    from pathlib import Path
    workspace = os.environ.get("WORKSPACE", str(Path(__file__).resolve().parents[1]))
    root = os.environ.get("AUTORAGBENCH_ROOT", workspace)
    model_dir = os.environ.get("MODEL_DIR", str(Path(workspace) / "models"))
    for key, replacement in (("@WORKSPACE@", workspace), ("@AUTORAGBENCH_ROOT@", root),
                             ("@MODEL_DIR@", model_dir), ("@BUILD_TMP@", os.environ.get("BUILD_TMP", "/tmp"))):
        value = value.replace(key, replacement)
    return value

_release_path("""df guard for @BUILD_TMP@.

Every 10 minutes: under 20 GiB sets warn, under 2 GiB sets immediate.
Callers exit after the current batch on warn, and before the next batch
on immediate. Quota headroom is not consulted.
""")
import subprocess
import threading

WORK_PATH = _release_path("@BUILD_TMP@")
WARN_BYTES = 20 * 1024 * 1024 * 1024
IMMEDIATE_BYTES = 2 * 1024 * 1024 * 1024


def avail_bytes():
    try:
        out = subprocess.check_output(
            ["df", "-B1", "--output=avail", WORK_PATH],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    try:
        return int(lines[-1])
    except ValueError:
        return None


class DiskGuard:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.level = ""
        self._stop = threading.Event()
        self._loop = None
        self._async_stop = None

    def bind(self, loop, async_stop):
        self._loop = loop
        self._async_stop = async_stop

    def start(self):
        if not self.enabled:
            return
        threading.Thread(target=self._loop_thread, name="disk-guard", daemon=True).start()

    def close(self):
        self._stop.set()

    def _loop_thread(self):
        while not self._stop.is_set():
            self.poll()
            if self.level == "immediate":
                self._trip_immediate()
                return
            if self.level == "warn":
                return
            self._stop.wait(600)

    def poll(self):
        if self.level:
            return
        avail = avail_bytes()
        if avail is None:
            print("DISK_GUARD level=unknown", flush=True)
            return
        if avail < IMMEDIATE_BYTES:
            self.level = "immediate"
            print(f"DISK_GUARD level=immediate avail_bytes={avail}", flush=True)
        elif avail < WARN_BYTES:
            self.level = "warn"
            print(f"DISK_GUARD level=warn avail_bytes={avail}", flush=True)

    def _trip_immediate(self):
        if self._loop is None or self._async_stop is None:
            return
        self._loop.call_soon_threadsafe(self._async_stop.set)

    def note_batch_end(self):
        """Called after a durable batch. Warn exits at this boundary."""
        if self.level == "warn":
            print("DISK_GUARD action=exit_after_batch", flush=True)
            return 75
        if self.level == "immediate":
            print("DISK_GUARD action=exit_immediate", flush=True)
            return 74
        return None
