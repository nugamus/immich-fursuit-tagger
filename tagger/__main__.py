"""`tagger` CLI. Subcommands are added milestone by milestone (see PLAN.md section 11)."""

import argparse
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def bench(args) -> int:
    from tagger.embedder import crop
    from tagger.runtime import Models

    models = Models(Path(args.models), device=args.device, threads=args.threads)
    images = [Image.open(p).convert("RGB") for p in sorted(Path(args.images).glob("*.jpg"))[: args.limit]]
    synthetic = not images
    if synthetic:
        # Inference time doesn't depend on content: random 1920x1440 previews, fixed head-sized crops.
        print(f"no .jpg images in {args.images}; using {args.limit} synthetic previews")
        rng = np.random.default_rng(0)
        images = [Image.fromarray(rng.integers(0, 255, (1440, 1920, 3), dtype=np.uint8)) for _ in range(args.limit)]

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
    if synthetic:
        crops = [img.crop((600, 300, 1000, 750)) for img in images]
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


def _context():
    from tagger import state
    from tagger.config import Config
    from tagger.immich import Immich

    cfg = Config()
    clients = {label: Immich(cfg.immich_url, key) for label, key in cfg.users().items()}
    return cfg, state.connect(cfg.db_path), clients


def scan(args) -> int:
    from tagger.runtime import Models
    from tagger.scan import scan_once

    cfg, db, clients = _context()
    models = Models(cfg.models_dir, device=cfg.inference_device, threads=cfg.threads, ttl_min=cfg.model_ttl_min)
    scan_once(cfg, db, models, clients, album_ids=args.album or None, limit=args.limit)
    return 0


def people(args) -> int:
    from tagger import people as ppl
    from tagger.scan import sync_users

    cfg, db, clients = _context()
    dry = cfg.dry_run or args.dry_run
    ids = sync_users(db, clients)
    if cfg.share_people or args.share:
        ppl.share_all(clients, ids, dry)
    for label, client in clients.items():
        print(label, ppl.tidy(db, label, client, (args.hide_below_pct or cfg.hide_background_pct) / 100, dry))
    return 0


def _write_pass(cfg, db, clients) -> str | None:
    """Guard, then one write pass (or a dry-run tally). Returns the read-only reason, if any."""
    from tagger.writer import Writer, guard

    reason = guard(cfg, clients)
    if reason:
        logging.getLogger("writer").error("READ-ONLY: %s", reason)
        return reason
    Writer(cfg, db, clients).sync()
    return None


def _people_pass(cfg, db, clients, ids) -> None:
    from tagger import people as ppl

    if cfg.share_people:
        ppl.share_all(clients, ids, cfg.dry_run)
    if cfg.human_thumbnails or cfg.hide_background_pct:
        for label, client in clients.items():
            ppl.tidy(db, label, client, cfg.hide_background_pct / 100, cfg.dry_run)


def write(args) -> int:
    from tagger.scan import sync_users
    from tagger.writer import paused

    cfg, db, clients = _context()
    sync_users(db, clients)
    if paused(db):
        print("paused (after undo); run `tagger resume` first")
        return 1
    return 2 if _write_pass(cfg, db, clients) else 0


def run(args) -> int:
    """The long-running loop: scan, write, housekeeping, sleep. Safe to restart at any time."""
    import json
    from datetime import datetime, timezone

    from tagger.runtime import Models
    from tagger.scan import scan_once, sync_users
    from tagger.writer import paused

    log = logging.getLogger("run")
    cfg, db, clients = _context()
    models = Models(cfg.models_dir, device=cfg.inference_device, threads=cfg.threads, ttl_min=cfg.model_ttl_min)
    from datetime import timedelta

    while True:
        pass_started = datetime.now(timezone.utc)
        reason = None
        try:
            if paused(db):
                log.info("paused (after undo); `tagger resume` to continue")
            else:
                scan_once(cfg, db, models, clients)
                reason = _write_pass(cfg, db, clients)
                _people_pass(cfg, db, clients, sync_users(db, clients))
            (cfg.data_dir / "health.json").write_text(json.dumps({
                "last_success": datetime.now(timezone.utc).isoformat(), "read_only": reason}))
        except Exception:  # keep the loop alive; health goes stale and the container turns unhealthy
            log.exception("pass failed")
        # Idle until photos change (cheap probe) or the periodic full pass is due. Models are only loaded when
        # there is something to process and are released after MODEL_TTL_MIN idle, freeing RAM for Immich.
        # Look back a bit before the pass started so clock skew can't hide an upload; known IDs are ignored.
        since = (pass_started - timedelta(minutes=10)).isoformat()
        deadline = time.monotonic() + cfg.scan_interval_min * 60
        while time.monotonic() < deadline:
            time.sleep(cfg.poll_interval_sec)
            models.release_if_idle()
            try:
                recent = {a for c in clients.values() for a in c.uploaded_since(since)}
            except Exception as e:  # Immich restarting etc.: just try again next poll
                log.debug("upload probe failed: %s", e)
                continue
            known = {r[0] for r in db.execute(f"SELECT asset_id FROM assets WHERE asset_id IN ({','.join('?' * len(recent))})",
                                              tuple(recent))} if recent else set()
            if recent - known:
                log.info("%d new photo(s) uploaded; starting a pass", len(recent - known))
                break


def health(args) -> int:
    """Docker HEALTHCHECK: healthy if a pass succeeded recently and writing isn't blocked by the version guard."""
    import json
    from datetime import datetime, timezone

    from tagger.config import Config

    cfg = Config()
    try:
        h = json.loads((cfg.data_dir / "health.json").read_text())
    except (OSError, ValueError):
        return 1
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(h["last_success"])).total_seconds()
    ok = age < cfg.scan_interval_min * 60 * 3 + 3600 and not h.get("read_only")
    print(h)
    return 0 if ok else 1


def undo(args) -> int:
    from tagger.writer import undo as run_undo

    cfg, db, clients = _context()
    print(run_undo(cfg, db, clients, since=args.since, user=args.user, dry_run=args.dry_run))
    if not args.dry_run:
        print("paused: the loop won't recreate anything until `tagger resume`")
    return 0


def resume(args) -> int:
    from tagger import state
    from tagger.config import Config

    cfg = Config()
    state.set_meta(state.connect(cfg.db_path), "paused", "0")
    print("resumed")
    return 0


def status(args) -> int:
    from tagger import state
    from tagger.config import Config

    cfg = Config()
    db = state.connect(cfg.db_path)
    q = lambda sql: db.execute(sql).fetchone()[0]  # noqa: E731
    print({"paused": state.get_meta(db, "paused") == "1",
           "assets": q("SELECT count(*) FROM assets"),
           "detections": q("SELECT count(*) FROM detections"),
           "assigned": q("SELECT count(*) FROM detections WHERE status = 'assigned'"),
           "rejected": q("SELECT count(*) FROM detections WHERE status = 'rejected'"),
           "faces_in_immich": q("SELECT count(*) FROM detections WHERE face_id IS NOT NULL"),
           "characters": q("SELECT count(*) FROM characters WHERE merged_into IS NULL"),
           "people_in_immich": q("SELECT count(*) FROM characters WHERE person_group_id IS NOT NULL"),
           "tagged_assets": q("SELECT count(*) FROM asset_tags")})
    return 0


def recluster(args) -> int:
    from tagger import state
    from tagger.config import Config
    from tagger.scan import recluster as run

    cfg = Config()
    print(run(state.connect(cfg.db_path), cfg))
    return 0


def report(args) -> int:
    from tagger import report as rep
    from tagger import state
    from tagger.config import Config

    cfg = Config()
    Path(args.out).write_text(rep.build(cfg, state.connect(cfg.db_path)), encoding="utf-8")
    print(f"wrote {args.out}")
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
    p = sub.add_parser("scan", help="one read-only pass: access scan, detect, embed, recognize (no Immich writes)")
    p.add_argument("--once", action="store_true", help="accepted for PLAN.md compatibility; scan always runs once")
    p.add_argument("--album", action="append", help="limit to this album ID (repeatable)")
    p.add_argument("--limit", type=int, help="process at most N assets")
    p.set_defaults(func=scan)
    p = sub.add_parser("people", help="human people housekeeping: sharpest thumbnails, hide background people, sharing")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--share", action="store_true", help="also share every user's people with the others")
    p.add_argument("--hide-below-pct", type=float, help="hide people whose largest face is narrower than this %% of the photo")
    p.set_defaults(func=people)
    p = sub.add_parser("recluster", help="dry run only: redo recognition from stored embeddings with current thresholds")
    p.set_defaults(func=recluster)
    p = sub.add_parser("run", help="long-running loop: scan, write to Immich, housekeeping")
    p.set_defaults(func=run)
    p = sub.add_parser("write", help="one write pass from the current state (respects DRY_RUN and the version guard)")
    p.set_defaults(func=write)
    p = sub.add_parser("undo", help="remove everything the tagger created in Immich, then pause")
    p.add_argument("--since", help="only things created at/after this ISO date")
    p.add_argument("--user", help="only this user label")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=undo)
    p = sub.add_parser("resume", help="clear the paused flag set by undo")
    p.set_defaults(func=resume)
    p = sub.add_parser("status", help="counts from the state database")
    p.set_defaults(func=status)
    p = sub.add_parser("health", help="exit 0 if healthy (for Docker HEALTHCHECK)")
    p.set_defaults(func=health)
    p = sub.add_parser("report", help="write the HTML dry-run report")
    p.add_argument("--out", default="/data/report.html")
    p.set_defaults(func=report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
