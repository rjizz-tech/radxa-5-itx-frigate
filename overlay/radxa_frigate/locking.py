"""Thread/process-safe AXCL arbitration, with preference for waiting detectors."""
import os
from pathlib import Path
import threading
import time
import uuid
from contextlib import contextmanager

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class AcceleratorBusy(TimeoutError):
    pass


class FileLock:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def acquire(self, deadline):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        while True:
            try:
                if os.name == "nt":
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    self.release()
                    raise AcceleratorBusy("AXCL inference queue wait exceeded")
                time.sleep(0.002)

    def release(self):
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
        except OSError:
            pass
        self.handle.close()
        self.handle = None


@contextmanager
def accelerator_slot(kind="face", timeout=0.020, directory=None):
    """A detector holds the admission gate while waiting and executing.

    Face jobs release the gate after acquiring the engine, so a waiting detector
    can enter and keep later face jobs from overtaking it. Every job is bounded.
    """
    directory = Path(directory or os.environ.get("RADXA_LOCK_DIR", "/dev/shm/radxa-axcl"))
    directory.mkdir(parents=True, exist_ok=True)
    pending = None
    if kind == 'object':
        # Advertise before acquiring admission: a newly arriving face must not
        # win the polling race ahead of an already queued detector.
        pending = directory/f'object-{os.getpid()}-{uuid.uuid4().hex}.pending'
        pending.touch(exist_ok=False)
    gate = FileLock(directory / "admission.lock")
    engine = FileLock(directory / "engine.lock")
    deadline = time.monotonic() + timeout
    try:
        gate.acquire(deadline)
        if kind != 'object':
            for marker in directory.glob('object-*.pending'):
                pid = int(marker.name.split('-')[1])
                if os.name != 'nt':
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        marker.unlink(missing_ok=True)
                        continue
                raise AcceleratorBusy('Object inference has priority')
        engine.acquire(deadline)
        if kind != "object":
            gate.release()
        yield
    finally:
        engine.release()
        gate.release()
        if pending is not None:
            pending.unlink(missing_ok=True)


class RateLimitedError:
    """Allow an error report once per interval without filling persistent logs."""
    def __init__(self, interval=60):
        self.interval = interval
        self.last = float("-inf")
        self.lock = threading.Lock()

    def due(self):
        with self.lock:
            now = time.monotonic()
            if now - self.last < self.interval:
                return False
            self.last = now
            return True
