import numpy as np
from PIL import Image

from tagger.detector import nms, postprocess
from tagger.embedder import crop


def test_postprocess_scales_filters_and_keeps_only_fursuit_class():
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.4, 0.4], [0.9, 0.9, 0.1, 0.1]], dtype=np.float32)
    logits = np.full((3, 2), -10.0, dtype=np.float32)
    logits[0, 0] = 3.0   # fursuit, confident
    logits[1, 1] = 3.0   # other slot: dropped
    logits[2, 0] = 0.0   # sigmoid 0.5 is not > 0.5: dropped
    out_boxes, scores = postprocess(boxes, logits, width=1000, height=500, min_score=0.5)
    assert len(scores) == 1
    np.testing.assert_allclose(out_boxes[0], [400, 150, 600, 350], atol=1e-3)
    # Box overhanging the top-left corner is clamped.
    logits[1] = [3.0, -10.0]
    out_boxes, _ = postprocess(boxes, logits, 1000, 500, 0.5)
    assert out_boxes.min() >= 0


def test_nms_suppresses_overlap_above_threshold():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 300, 300]], dtype=np.float32)
    scores = np.array([0.9, 0.95, 0.8], dtype=np.float32)
    assert list(nms(boxes, scores, 0.1)) == [1, 2]


def test_crop_floor_ceil_clamp_and_min_size():
    img = Image.new("RGB", (200, 100))
    assert crop(img, (10.7, 5.2, 90.1, 80.9)).size == (81, 76)
    assert crop(img, (-20, -20, 300, 300)).size == (200, 100)
    assert crop(img, (0, 0, 63.5, 100)) is not None  # ceil(63.5) = 64
    assert crop(img, (0, 0, 63, 100)) is None
