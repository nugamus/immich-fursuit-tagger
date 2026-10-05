"""Fursuit head detector: RF-DETR Medium exported to ONNX.

Pre/post-processing is a faithful port of rfdetr 1.8.3 `RFDETR.predict` + `PostProcess`
and the NMS in aibyou0830/immich-ml-furry `fursuit_recognition/detection.py` (AGPL-3.0).
"""

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
NUM_SELECT = 300
FURSUIT_CLASS = 0


def preprocess(img: Image.Image, resolution: int) -> np.ndarray:
    """RGB image -> (1, 3, res, res) float32. Plain stretch, no letterbox (matches torchvision resize)."""
    # Resize each channel in float ("F" mode): resizing uint8 rounds and breaks parity with torch's float resize.
    channels = np.asarray(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
    chw = np.stack([np.asarray(Image.fromarray(c, mode="F").resize((resolution, resolution), Image.Resampling.BILINEAR))
                    for c in channels])
    return ((chw - MEAN) / STD)[None].astype(np.float32)


def postprocess(boxes_cxcywh: np.ndarray, logits: np.ndarray, width: int, height: int,
                min_score: float) -> tuple[np.ndarray, np.ndarray]:
    """One image's raw outputs (Q, 4), (Q, C) -> xyxy pixel boxes and scores, best first."""
    num_classes = logits.shape[1]
    prob = 1.0 / (1.0 + np.exp(-logits.reshape(-1).astype(np.float64)))
    top = np.argsort(-prob, kind="stable")[:NUM_SELECT]
    scores, queries, labels = prob[top], top // num_classes, top % num_classes

    cx, cy, w, h = boxes_cxcywh[queries].T.astype(np.float64)
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) * [width, height, width, height]
    boxes = np.clip(boxes, 0, [width, height, width, height])

    keep = (scores > min_score) & (labels == FURSUIT_CLASS)
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


def iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    areas = np.maximum(0.0, (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]))
    return inter / np.maximum(area + areas - inter, 1e-12)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        order = rest[iou(boxes[current], boxes[rest]) <= iou_threshold]
    return np.asarray(keep, dtype=np.intp)


class Detector:
    def __init__(self, session, min_score: float = 0.5, nms_iou: float = 0.1):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.resolution = session.get_inputs()[0].shape[-1]
        self.min_score = min_score
        self.nms_iou = nms_iou

    def __call__(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        dets, logits = self.session.run(["dets", "labels"], {self.input_name: preprocess(img, self.resolution)})
        boxes, scores = postprocess(dets[0], logits[0], img.width, img.height, self.min_score)
        keep = nms(boxes, scores, self.nms_iou)
        return boxes[keep], scores[keep]
