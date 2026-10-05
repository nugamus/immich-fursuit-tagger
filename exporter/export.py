"""One-shot model exporter: published PyTorch checkpoints -> ONNX + models/manifest.json.

No-op when the manifest already matches the pinned sources. Runs as a compose init service.
Detector and embedder are versioned separately so an embedder-only change triggers re-embedding, not re-detection.
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("exporter")

# Bump when export code changes in a way that alters model outputs.
EXPORT_CODE_VERSION = 1

DETECTOR_REPO = "aibyou0830/rf-detr-for-fur"
DETECTOR_REVISION = os.environ.get("DETECTOR_REVISION", "ce47a3d3e9910638ff8ea66c18d7fe6f6f374d77")
HEAD_REPO = "aibyou0830/supcon-multiview-for-fur"
HEAD_REVISION = os.environ.get("HEAD_REVISION", "8b3fd67803e9219b5573d7cf6d1f7894edb9d7e4")
BACKBONE_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"
BACKBONE_REVISION = os.environ.get("BACKBONE_REVISION", "main")

DETECTOR_RESOLUTION = 576
EMBEDDER_RESOLUTION = 512
OPSET = 17


def version_of(*parts: object) -> str:
    return hashlib.sha256(json.dumps([EXPORT_CODE_VERSION, *parts]).encode()).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_backbone_revision() -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(BACKBONE_REPO, revision=BACKBONE_REVISION).sha


def export_detector(out_dir: Path, version: str) -> dict:
    from huggingface_hub import hf_hub_download
    from rfdetr import RFDETRMedium

    ckpt = hf_hub_download(DETECTOR_REPO, "checkpoint_best_total.pth", revision=DETECTOR_REVISION)
    model = RFDETRMedium(pretrain_weights=ckpt, num_classes=1, device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        exported = model.export(output_dir=tmp, opset_version=OPSET, verbose=False)
        target = out_dir / f"detector-{version}.onnx"
        shutil.move(str(exported), target)
    return {"version": version, "file": target.name, "sha256": sha256_file(target),
            "source": f"{DETECTOR_REPO}@{DETECTOR_REVISION}", "resolution": DETECTOR_RESOLUTION,
            "outputs": {"dets": "cxcywh normalized", "labels": "logits, class 0 = fursuit"}}


def build_projector(state_dict):
    from torch import nn

    weights = sorted(int(k.split(".")[1]) for k in state_dict if k.startswith("head.") and k.endswith(".weight"))
    dims = [state_dict[f"head.{weights[0]}.weight"].shape[1]] + [state_dict[f"head.{i}.weight"].shape[0] for i in weights]
    layers: list = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.Tanh(), nn.Dropout(0.3)]
    projector = nn.Sequential(*layers)
    projector.load_state_dict({k.removeprefix("head."): v for k, v in state_dict.items()})
    return projector.eval()


def export_embedder(out_dir: Path, version: str, backbone_revision: str) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import DINOv3ViTModel

    class Embedder(torch.nn.Module):
        """DINOv3 backbone + SupCon head + L2 norm, fused into one graph."""

        def __init__(self, backbone, projector):
            super().__init__()
            self.backbone, self.projector = backbone, projector

        def forward(self, pixel_values):
            out = self.backbone(pixel_values=pixel_values)
            features = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
            return torch.nn.functional.normalize(self.projector(features), p=2, dim=-1)

    backbone = DINOv3ViTModel.from_pretrained(BACKBONE_REPO, revision=backbone_revision).eval()
    head = hf_hub_download(HEAD_REPO, "supcon_projection_head_best.pth", revision=HEAD_REVISION)
    model = Embedder(backbone, build_projector(torch.load(head, map_location="cpu", weights_only=True))).eval()

    target = out_dir / f"embedder-{version}.onnx"
    dummy = torch.zeros(1, 3, EMBEDDER_RESOLUTION, EMBEDDER_RESOLUTION)
    with torch.inference_mode():
        torch.onnx.export(model, (dummy,), str(target), input_names=["pixel_values"], output_names=["embedding"],
                          dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
                          opset_version=OPSET, dynamo=False)
    return {"version": version, "file": target.name, "sha256": sha256_file(target),
            "source": f"{BACKBONE_REPO}@{backbone_revision} + {HEAD_REPO}@{HEAD_REVISION}",
            "resolution": EMBEDDER_RESOLUTION, "dim": 512, "license": "DINOv3 License. Built with DINOv3."}


def is_current(entry: dict | None, version: str, out_dir: Path) -> bool:
    return bool(entry) and entry["version"] == version and (out_dir / entry["file"]).exists()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(os.environ.get("MODELS_DIR", "/models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": 1}

    det_version = version_of(DETECTOR_REPO, DETECTOR_REVISION, DETECTOR_RESOLUTION)
    if is_current(manifest.get("detector"), det_version, out_dir):
        log.info("detector %s up to date", det_version)
    else:
        log.info("exporting detector %s", det_version)
        manifest["detector"] = export_detector(out_dir, det_version)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    try:
        backbone_revision = resolve_backbone_revision()
        emb_version = version_of(BACKBONE_REPO, backbone_revision, HEAD_REPO, HEAD_REVISION, EMBEDDER_RESOLUTION)
        if is_current(manifest.get("embedder"), emb_version, out_dir):
            log.info("embedder %s up to date", emb_version)
        else:
            log.info("exporting embedder %s", emb_version)
            manifest["embedder"] = export_embedder(out_dir, emb_version, backbone_revision)
            manifest_path.write_text(json.dumps(manifest, indent=2))
    except OSError as e:  # gated repo without access, or offline
        log.error("embedder export failed (HF_TOKEN set and DINOv3 license approved?): %s", str(e).splitlines()[0])
    # Remove stale exports so old weights don't pile up on the N100's disk.
    live = {manifest[k]["file"] for k in ("detector", "embedder") if k in manifest}
    for stale in out_dir.glob("*.onnx"):
        if stale.name not in live:
            stale.unlink()
    return 0 if "embedder" in manifest and "detector" in manifest else 1


if __name__ == "__main__":
    sys.exit(main())
