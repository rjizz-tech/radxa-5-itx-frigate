"""Decode AXERA's six-tensor YOLO26 export, not the Ultralytics e2e export.

Reference: AXERA-TECH/axcl-samples examples/base/detection.hpp,
generate_proposals_yolo26. See THIRD_PARTY_NOTICES.md.
"""
import numpy as np


def nms(boxes, scores, threshold=0.45, limit=20):
    """Class-agnostic NMS matching AXERA's reference get_out_bbox."""
    order = np.argsort(-scores, kind="stable")
    selected = []
    while order.size and len(selected) < limit:
        current = int(order[0])
        selected.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        lt = np.maximum(boxes[current, :2], boxes[remaining, :2])
        rb = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        intersection = np.prod(np.maximum(0, rb - lt), axis=1)
        area = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
        union = area[current] + area[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= threshold]
    return selected


def decode(outputs, width=640, height=640, confidence=0.45, iou=0.45):
    if len(outputs) != 6:
        raise ValueError("AXERA YOLO26 requires six raw box/class tensors")
    proposals, probabilities, labels = [], [], []
    for level, stride in enumerate((8, 16, 32)):
        h, w = height // stride, width // stride
        box, cls = (np.asarray(outputs[level * 2 + offset]) for offset in (0, 1))
        if box.shape not in ((1, h, w, 4), (1, h*w, 4), (h*w, 4)):
            raise ValueError(f"YOLO26 box tensor at stride {stride} has unsupported layout {box.shape}")
        if cls.shape not in ((1, h, w, 80), (1, h*w, 80), (h*w, 80)):
            raise ValueError(f"YOLO26 class tensor at stride {stride} has unsupported layout {cls.shape}")
        if box.dtype.kind != 'f' or cls.dtype.kind != 'f':
            raise ValueError("YOLO26 outputs must be dequantized floats")
        if not np.isfinite(box).all() or not np.isfinite(cls).all():
            raise ValueError("YOLO26 output contains non-finite values")
        box, cls = box.reshape(-1, 4), cls.reshape(-1, 80)
        classes = cls.argmax(axis=1)
        logits = cls[np.arange(len(cls)), classes]
        scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        keep = scores >= confidence
        indices = np.flatnonzero(keep)
        centers = np.column_stack(((indices % w + 0.5)*stride, (indices // w + 0.5)*stride))
        distances = box[keep]*stride
        corners = np.column_stack((centers - distances[:, :2], centers + distances[:, 2:]))
        corners[:, (0, 2)] = np.clip(corners[:, (0, 2)], 0, width - 1)
        corners[:, (1, 3)] = np.clip(corners[:, (1, 3)], 0, height - 1)
        valid = np.all(corners[:, 2:] > corners[:, :2], axis=1)
        proposals.append(corners[valid])
        probabilities.append(scores[keep][valid])
        labels.append(classes[keep][valid])
    boxes = np.concatenate(proposals)
    scores = np.concatenate(probabilities)
    classes = np.concatenate(labels)
    result = np.zeros((20, 6), dtype=np.float32)
    for row, index in enumerate(nms(boxes, scores, iou)):
        x0, y0, x1, y1 = boxes[index]
        result[row] = [classes[index], scores[index], y0/height, x0/width, y1/height, x1/width]
    return result
