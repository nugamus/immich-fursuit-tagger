"""`tagger` CLI. Subcommands are added milestone by milestone (see PLAN.md section 11)."""

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path

from PIL import Image


def bench(args) -> int:
    from tagger.embedder import crop
    from tagger.runtime import Models

    models = Models(Path(args.models), device=args.device, threads=args.threads)
    images = [Image.open(p).convert("RGB") for p in sorted(Path(args.images).glob("*.jpg"))[: args.limit]]
    if not images:
        print(f"no .jpg images in {args.images}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    detector = models.detector()
    print(f"detector load {time.perf_counter() - t0:.2f}s")
    detector(images[0])  # warm-up (OpenVINO compiles on first run)

    det_ms, crops = [], []
    for img in images:
        t = time.perf_counter()
        boxes, _ = detector(img)
        det_ms.append((time.perf_counter() - t) * 1000)
        crops += [c for c in (crop(img, b) for b in boxes) if c is not None]
    print(f"detect: {len(images)} images, median {statistics.median(det_ms):.0f} ms, max {max(det_ms):.0f} ms, "
          f"{len(crops)} crops")

    if models.embedder_version and crops:
        t0 = time.perf_counter()
        embedder = models.embedder()
        print(f"embedder load {time.perf_counter() - t0:.2f}s")
        embedder(crops[:1])
        t = time.perf_counter()
        embedder(crops)
        print(f"embed: {len(crops)} crops, {(time.perf_counter() - t) * 1000 / len(crops):.0f} ms/crop")
    else:
        print("embed: skipped (no embedder in manifest)")

    try:
        import resource  # Unix only
        print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    except ImportError:
        pass
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="tagger")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bench", help="time detection and embedding on sample images")
    p.add_argument("--images", default=os.environ.get("BENCH_DIR", "/data/bench"))
    p.add_argument("--models", default=os.environ.get("MODELS_DIR", "/models"))
    p.add_argument("--device", default=os.environ.get("INFERENCE_DEVICE", "auto"))
    p.add_argument("--threads", type=int, default=int(os.environ["THREADS"]) if os.environ.get("THREADS") else None)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=bench)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
