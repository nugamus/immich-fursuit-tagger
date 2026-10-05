"""ONNX Runtime sessions: execution-provider selection, lazy load and idle unload."""

import json
import logging
import os
import threading
import time
from pathlib import Path

import onnxruntime as ort

from tagger.detector import Detector
from tagger.embedder import Embedder

log = logging.getLogger("runtime")

# Preference order for INFERENCE_DEVICE=auto.
AUTO_ORDER = ["cuda", "rocm", "openvino", "cpu"]
PROVIDER = {
    "cuda": "CUDAExecutionProvider",
    "rocm": "MIGraphXExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def resolve_device(requested: str) -> str:
    available = set(ort.get_available_providers())
    if requested != "auto":
        if PROVIDER[requested] not in available:
            raise RuntimeError(f"INFERENCE_DEVICE={requested} but ORT only has {sorted(available)}")
        return requested
    return next(d for d in AUTO_ORDER if PROVIDER[d] in available)


def provider_list(device: str) -> list:
    if device == "openvino":
        # The iGPU needs /dev/dri passed through; fall back to OpenVINO on CPU otherwise.
        gpu = Path("/dev/dri").exists()
        options = {"device_type": "GPU" if gpu else "CPU", "precision": "FP16" if gpu else "FP32",
                   # Compiling for the iGPU takes ~30 s on an N100; cache it so idle unload/reload stays cheap.
                   "cache_dir": os.environ.get("OV_CACHE_DIR", "/data/ov_cache")}
        return [(PROVIDER[device], options), PROVIDER["cpu"]]
    return [PROVIDER[device]] if device == "cpu" else [PROVIDER[device], PROVIDER["cpu"]]


class Models:
    """Detector + embedder sessions, loaded on first use and released after `ttl_min` idle minutes."""

    def __init__(self, models_dir: Path, device: str = "auto", threads: int | None = None, ttl_min: float = 10):
        self.models_dir = Path(models_dir)
        self.manifest = json.loads((self.models_dir / "manifest.json").read_text())
        self.device = resolve_device(device)
        self.threads = threads or max(1, (os.cpu_count() or 2) - 1)
        self.ttl = ttl_min * 60
        self._lock = threading.Lock()
        self._detector: Detector | None = None
        self._embedder: Embedder | None = None
        self._last_used = 0.0
        log.info("inference device %s (%s), %d threads", self.device, provider_list(self.device)[0], self.threads)

    @property
    def detector_version(self) -> str:
        return self.manifest["detector"]["version"]

    @property
    def embedder_version(self) -> str | None:
        return self.manifest.get("embedder", {}).get("version")

    def _session(self, kind: str) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        path = self.models_dir / self.manifest[kind]["file"]
        return ort.InferenceSession(str(path), sess_options=opts, providers=provider_list(self.device))

    def detector(self) -> Detector:
        with self._lock:
            self._last_used = time.monotonic()
            if self._detector is None:
                self._detector = Detector(self._session("detector"))
            return self._detector

    def embedder(self) -> Embedder:
        with self._lock:
            self._last_used = time.monotonic()
            if self._embedder is None:
                if "embedder" not in self.manifest:
                    raise RuntimeError("embedder missing from manifest; run the exporter with DINOv3 access")
                self._embedder = Embedder(self._session("embedder"))
            return self._embedder

    def release_if_idle(self) -> bool:
        """Drop sessions after TTL without work. Called from the scheduler loop."""
        with self._lock:
            loaded = self._detector is not None or self._embedder is not None
            if loaded and time.monotonic() - self._last_used > self.ttl:
                self._detector = self._embedder = None
                log.info("released models after %.0f min idle", self.ttl / 60)
                return True
            return False
