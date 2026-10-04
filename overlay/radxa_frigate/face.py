"""AXCL inference adapters retaining Frigate's native face classifier and UI."""
import functools
import logging
import time
import cv2
import numpy as np

from .locking import AcceleratorBusy, RateLimitedError
from .runtime import ModelSession
from .yolo26 import nms

logger = logging.getLogger(__name__)
_error_limiter = RateLimitedError()


def decode_yunet(outputs, size=640, threshold=0.5):
    if len(outputs) != 12:
        raise ValueError("YuNet requires cls/obj/bbox/kps outputs at three strides")
    all_rows = []
    for level, stride in enumerate((8, 16, 32)):
        count = (size // stride) ** 2
        for group, channels in enumerate((1, 1, 4, 10)):
            if np.asarray(outputs[group * 3 + level]).shape != (1, count, channels):
                raise ValueError("YuNet output shape does not match the pinned export")
        cls = np.asarray(outputs[level]).reshape(-1)
        obj = np.asarray(outputs[3 + level]).reshape(-1)
        boxes = np.asarray(outputs[6 + level]).reshape(-1, 4)
        keypoints = np.asarray(outputs[9 + level]).reshape(-1, 10)
        scores = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        ids = np.flatnonzero(scores >= threshold)
        if not ids.size:
            continue
        grid = np.column_stack((ids % (size // stride), ids // (size // stride)))
        centers = (grid + boxes[ids, :2]) * stride
        dimensions = np.exp(np.clip(boxes[ids, 2:], -20, 20)) * stride
        origins = centers - dimensions / 2
        points = keypoints[ids].reshape(-1, 5, 2)
        points = (points + grid[:, None, :]) * stride
        all_rows.append(np.column_stack((origins, dimensions, points.reshape(-1, 10), scores[ids])))
    if not all_rows:
        return np.empty((0, 15), np.float32)
    rows = np.concatenate(all_rows).astype(np.float32)
    corners = np.column_stack((rows[:, :2], rows[:, :2] + rows[:, 2:4]))
    return rows[nms(corners, rows[:, -1], threshold=0.3, limit=500)]


def yunet_input(image, size=640):
    """Black letterbox of raw BGR pixels; no normalization, float NCHW."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Empty face image")
    ratio = min(size / w, size / h)
    resized_w, resized_h = max(1, int(w*ratio)), max(1, int(h*ratio))
    x, y = (size-resized_w)//2, (size-resized_h)//2
    canvas = np.zeros((size, size, 3), np.float32)
    canvas[y:y+resized_h, x:x+resized_w] = cv2.resize(image, (resized_w, resized_h))
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
    return tensor, (ratio, x, y)


class AxclFaceDetector:
    """Subset of cv2.FaceDetectorYN used by Frigate, with AXCL inference."""
    def __init__(self, config):
        self.session = ModelSession(config.axcl_manifest, "yunet", timeout=config.axcl_wait_ms/1000)

    def setInputSize(self, size):
        # AXCL graph is static; detect maps back to the actual crop size.
        pass

    def detect(self, image):
        tensor, (ratio, x, y) = yunet_input(image)
        rows = decode_yunet(self.session.run(tensor))
        if not rows.size:
            return 0, None
        rows[:, 0] = (rows[:, 0]-x)/ratio
        rows[:, 1] = (rows[:, 1]-y)/ratio
        rows[:, 2:4] /= ratio
        points = rows[:, 4:14].reshape(-1, 5, 2)
        points[..., 0] = (points[..., 0]-x)/ratio
        points[..., 1] = (points[..., 1]-y)/ratio
        # Clip before upstream's uint16 cast, which otherwise wraps negatives.
        h, w = image.shape[:2]
        far = rows[:, :2] + rows[:, 2:4]
        rows[:, 0] = np.clip(rows[:, 0], 0, w-1)
        rows[:, 1] = np.clip(rows[:, 1], 0, h-1)
        far[:, 0] = np.clip(far[:, 0], 0, w)
        far[:, 1] = np.clip(far[:, 1], 0, h)
        rows[:, 2:4] = far-rows[:, :2]
        rows = rows[np.all(rows[:, 2:4] > 0, axis=1)]
        return len(rows), rows if len(rows) else None


def create_arcface_embedder(config):
    # Import lazily to avoid a cycle while Frigate imports its recognizers.
    from frigate.embeddings.onnx.face_embedding import ArcfaceEmbedding

    class AxclArcFaceEmbedding(ArcfaceEmbedding):
        def __init__(self):
            self.config = config
            self.session = ModelSession(config.axcl_manifest, "arcface", timeout=config.axcl_wait_ms/1000)

        def __call__(self, inputs):
            # Reuse Frigate's RGB/resize/padding/[-1,1] preprocessing exactly.
            tensor = self._preprocess_inputs(inputs)[0]["data"]
            result = self.session.run(tensor)
            if len(result) != 1 or result[0].shape != (1, 512):
                raise ValueError("ArcFace must return one 512-component embedding")
            if np.linalg.norm(result[0]) < 1e-8:
                raise ValueError("ArcFace returned a zero embedding")
            return result

    return AxclArcFaceEmbedding()


def guarded_face_call(method):
    """Contain unavailable-model/AXCL errors without killing video processing."""
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        if self.config.face_recognition.backend != "axcl":
            return method(self, *args, **kwargs)
        try:
            result = method(self, *args, **kwargs)
            from .status import report
            report('face')
            return result
        except AcceleratorBusy:
            # A face can be skipped; object processing retains priority.
            if method.__name__ == "handle_request":
                return {"success": False, "message": "AX-M1 busy; retry enrollment shortly."}
            return None
        except Exception as exc:
            if _error_limiter.due():
                logger.error("AXCL face recognition unavailable: %s", exc)
            from .status import report
            report("face", error=str(exc))
            if method.__name__ == "handle_request":
                return {"success": False, "message": "AXCL face backend unavailable; inspect hardware/model health."}
            return None
    return wrapped


def safe_embedding(embedder, image):
    """Used by native background library rebuilding, which must not hang on errors."""
    try:
        return embedder([image])[0].squeeze()
    except Exception as exc:
        if _error_limiter.due():
            logger.warning("Skipping face library sample: %s", exc)
        return None


def safe_library_embedding(recognizer, image):
    """Retry busy enrollment work briefly, without unbounded queues or starvation."""
    deadline = time.monotonic()+2
    try:
        aligned = recognizer.align_face(image, image.shape[1], image.shape[0])
        while True:
            try:
                embedding = recognizer.face_embedder([aligned])[0].squeeze()
                from .status import report
                report('library')
                return embedding
            except AcceleratorBusy:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
    except Exception as exc:
        if _error_limiter.due():
            logger.warning('Face library sample could not be embedded: %s', exc)
        from .status import report
        report('library', error=str(exc))
        return None
