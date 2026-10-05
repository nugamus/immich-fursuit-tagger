"""ONNX vs PyTorch parity (M1). Needs the exporter extra plus exported models; skipped otherwise.

    MODELS_DIR=work/models SAMPLES_DIR=work/samples CKPT_DIR=work/ckpt uv run pytest tests/test_parity.py -s
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tagger.detector import Detector, iou, nms

MODELS = Path(os.environ.get("MODELS_DIR", "work/models"))
SAMPLES = sorted(Path(os.environ.get("SAMPLES_DIR", "work/samples")).glob("*.jpg"))
CKPT = Path(os.environ.get("CKPT_DIR", "work/ckpt"))

pytestmark = pytest.mark.skipif(not SAMPLES or not (MODELS / "manifest.json").exists(), reason="no models/samples")


@pytest.fixture(scope="module")
def manifest():
    return json.loads((MODELS / "manifest.json").read_text())


def test_detector_parity(manifest):
    ort = pytest.importorskip("onnxruntime")
    rfdetr = pytest.importorskip("rfdetr")
    torch_model = rfdetr.RFDETRMedium(pretrain_weights=str(CKPT / "checkpoint_best_total.pth"), num_classes=1, device="cpu")
    onnx_model = Detector(ort.InferenceSession(str(MODELS / manifest["detector"]["file"])), min_score=0.3)

    worst_iou, worst_score, compared = 1.0, 0.0, 0
    for path in SAMPLES:
        img = Image.open(path).convert("RGB")
        ref = torch_model.predict(img, threshold=0.3)
        ref = ref[np.array([n == "fursuit" for n in ref.data["class_name"]], dtype=bool)]
        ref_boxes, ref_scores = ref.xyxy.astype(np.float32), ref.confidence.astype(np.float32)
        keep = nms(ref_boxes, ref_scores, 0.1)
        ref_boxes, ref_scores = ref_boxes[keep], ref_scores[keep]

        boxes, scores = onnx_model(img)
        # Borderline detections near the threshold may flip; only compare confident ones.
        confident = ref_scores >= 0.4
        for box, score in zip(ref_boxes[confident], ref_scores[confident]):
            assert len(boxes), f"{path.name}: ONNX missed a detection with score {score:.3f}"
            ious = iou(box, boxes)
            j = int(np.argmax(ious))
            worst_iou = min(worst_iou, float(ious[j]))
            worst_score = max(worst_score, abs(float(scores[j]) - float(score)))
            compared += 1
        print(f"{path.name}: torch {len(ref_boxes)} onnx {len(boxes)}")

    print(f"compared {compared} boxes, worst IoU {worst_iou:.4f}, worst score diff {worst_score:.4f}")
    assert compared > 0
    assert worst_iou >= 0.98
    assert worst_score <= 0.02


def test_embedder_parity(manifest):
    """Reference = the fork's PyTorch path (torchvision transform + DINOv3 + SupCon head)."""
    if "embedder" not in manifest:
        pytest.skip("embedder not exported (DINOv3 access?)")
    ort = pytest.importorskip("onnxruntime")
    torch = pytest.importorskip("torch")
    from torchvision import transforms
    from transformers import DINOv3ViTModel

    from exporter.export import BACKBONE_REPO, build_projector
    from tagger.detector import Detector
    from tagger.embedder import Embedder, crop

    backbone = DINOv3ViTModel.from_pretrained(BACKBONE_REPO).eval()
    projector = build_projector(torch.load(CKPT / "supcon_projection_head_best.pth", map_location="cpu", weights_only=True))
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    detector = Detector(ort.InferenceSession(str(MODELS / manifest["detector"]["file"])))
    embedder = Embedder(ort.InferenceSession(str(MODELS / manifest["embedder"]["file"])))

    worst = 1.0
    for path in SAMPLES:
        img = Image.open(path).convert("RGB")
        crops = [c for c in (crop(img, b) for b in detector(img)[0]) if c is not None]
        if not crops:
            continue
        with torch.inference_mode():
            out = backbone(pixel_values=torch.stack([transform(c) for c in crops]))
            feats = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
            ref = torch.nn.functional.normalize(projector(feats), p=2, dim=-1).numpy()
        got = embedder(crops)
        worst = min(worst, float((ref * got).sum(axis=1).min()))
    print(f"embedder worst cosine similarity {worst:.5f}")
    assert worst >= 0.999
