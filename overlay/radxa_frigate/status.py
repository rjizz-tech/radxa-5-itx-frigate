"""Small, rate-limited operational state files on RAM, never on recording disks."""
import json
import os
from pathlib import Path
import threading
import time

_last = {}
_lock = threading.Lock()


def report(component, error=None):
    with _lock:
        now = time.time()
        key = (component, error)
        if now-_last.get(key, 0) < 10:
            return
        _last[key] = now
        directory = Path(os.environ.get("RADXA_STATE_DIR", "/dev/shm/radxa-status"))
        try:
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{component}.json"
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({"timestamp": now, "error": error}))
            temporary.replace(target)
        except OSError:
            pass
