"""Crop quality in [0, 1]: geometric mean of detector confidence, crop size and sharpness."""

import numpy as np
from PIL import Image

FULL_SIZE_PX = 256      # shorter side at which size stops mattering
SHARPNESS_HALF = 100.0  # Laplacian variance that maps to 0.5 (measured on a 224x224 grayscale crop)


def sharpness(crop: Image.Image) -> float:
    g = np.asarray(crop.convert("L").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def quality(score: float, crop: Image.Image) -> float:
    size = min(1.0, min(crop.size) / FULL_SIZE_PX)
    s = sharpness(crop)
    sharp = s / (s + SHARPNESS_HALF)
    return float((score * size * sharp) ** (1 / 3))
