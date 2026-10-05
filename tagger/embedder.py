"""Fursuit identity embedder: DINOv3 ViT-S/16 + SupCon head + L2 norm, fused into one ONNX graph.

Cropping and preprocessing port aibyou0830/immich-ml-furry `fursuit_recognition/recognition.py` (AGPL-3.0).
Built with DINOv3.
"""

import math

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
MIN_CROP = 64


def crop(img: Image.Image, box) -> Image.Image | None:
    """floor/ceil the box, clamp to the image, drop crops whose shorter side is under MIN_CROP."""
    x1, y1, x2, y2 = (float(v) for v in box)
    left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
    right, bottom = min(img.width, math.ceil(x2)), min(img.height, math.ceil(y2))
    if right <= left or bottom <= top or min(right - left, bottom - top) < MIN_CROP:
        return None
    return img.crop((left, top, right, bottom))


def preprocess(crops: list[Image.Image], resolution: int) -> np.ndarray:
    # torchvision Resize on a PIL image is PIL's uint8 bilinear resize, so this matches the reference exactly.
    batch = [np.asarray(c.convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR),
                        dtype=np.float32).transpose(2, 0, 1) / 255.0 for c in crops]
    return ((np.stack(batch) - MEAN) / STD).astype(np.float32)


class Embedder:
    def __init__(self, session, batch_size: int = 8):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.resolution = session.get_inputs()[0].shape[-1]
        self.batch_size = batch_size

    def __call__(self, crops: list[Image.Image]) -> np.ndarray:
        """Crops -> (N, 512) L2-normalized float32 embeddings."""
        out = [self.session.run(None, {self.input_name: preprocess(crops[i:i + self.batch_size], self.resolution)})[0]
               for i in range(0, len(crops), self.batch_size)]
        return np.concatenate(out) if out else np.empty((0, 512), dtype=np.float32)
