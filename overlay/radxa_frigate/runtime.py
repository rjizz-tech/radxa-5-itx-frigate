"""Checksum-verified, explicitly PCIe-only AXEngine sessions."""
import hashlib
import json
from pathlib import Path
import threading
import time
import numpy as np

from .locking import accelerator_slot
from .status import report


def read_manifest(path):
    document = json.loads(Path(path).read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported AXCL manifest schema")
    return document


class ModelSession:
    def __init__(self, manifest_path, model_id, kind="face", timeout=0.020):
        self.manifest_path = Path(manifest_path)
        self.model_id = model_id
        self.kind = kind
        self.timeout = timeout
        self.session = None
        self.spec = None
        self.mutex = threading.Lock()
        self.load_retry_at = 0
        self.load_error = None

    def _load(self):
        document = read_manifest(self.manifest_path)
        spec = document["models"][self.model_id]
        model = (self.manifest_path.parent / spec["file"]).resolve()
        if not model.is_relative_to(self.manifest_path.parent.resolve()):
            raise ValueError("Model must be inside its manifest directory")
        if not model.is_file():
            raise FileNotFoundError(f"Missing compiled {self.model_id} model: {model}")
        with model.open("rb") as handle:
            sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if sha != spec["sha256"]:
            raise ValueError(f"Checksum mismatch for {self.model_id}")
        import axengine as axe
        provider = "AXCLRTExecutionProvider"
        if provider not in axe.get_available_providers():
            raise RuntimeError("PCIe AXCL provider unavailable; CPU/onboard-NPU fallback is forbidden")
        session = axe.InferenceSession(str(model), providers=[provider])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Expected one AXCL model input")
        shape = list(inputs[0].shape)
        if shape != spec["input"]["shape"]:
            raise ValueError(f"{self.model_id} input shape {shape} differs from manifest")
        # pyAXEngine exposes numpy dtype objects, not ONNX type strings.
        if np.dtype(inputs[0].dtype) != np.dtype(spec["input"]["dtype"]):
            raise ValueError(f"{self.model_id} input dtype differs from manifest")
        self.session, self.spec = session, spec

    def run(self, tensor):
        # The thread lock protects lazy session creation and native buffers.
        if not self.mutex.acquire(timeout=self.timeout):
            from .locking import AcceleratorBusy
            raise AcceleratorBusy("AXCL session busy")
        try:
            with accelerator_slot(self.kind, self.timeout):
                if self.session is None:
                    if time.monotonic() < self.load_retry_at:
                        raise RuntimeError(f'{self.model_id} initialization retry deferred: {self.load_error}')
                    try:
                        self._load()
                    except Exception as exc:
                        self.load_error = str(exc)
                        self.load_retry_at = time.monotonic()+10
                        raise
                array = np.asarray(tensor)
                if list(array.shape) != self.spec["input"]["shape"]:
                    raise ValueError(f"{self.model_id} tensor shape mismatch")
                if array.dtype != np.dtype(self.spec["input"]["dtype"]):
                    raise ValueError(f"{self.model_id} tensor dtype mismatch")
                outputs = self.session.run(None, {self.session.get_inputs()[0].name: np.ascontiguousarray(array)})
                names = self.spec.get("output_names")
                if names:
                    actual = [item.name for item in self.session.get_outputs()]
                    if set(actual) != set(names) or len(actual) != len(names):
                        raise ValueError(f"{self.model_id} output names differ from manifest")
                    # Fetch native order, then reorder locally: AXEngine 0.1.3 does not
                    # safely implement reordered output_names in session.run.
                    by_name = dict(zip(actual, outputs, strict=True))
                    outputs = [by_name[name] for name in names]
                if not all(np.isfinite(output).all() for output in outputs):
                    raise ValueError(f"{self.model_id} returned non-finite values")
                report(self.model_id)
                return outputs
        except Exception as exc:
            from .locking import AcceleratorBusy
            if not isinstance(exc, AcceleratorBusy):
                report(self.model_id, error=str(exc))
            raise
        finally:
            self.mutex.release()
