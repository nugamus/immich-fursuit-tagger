"""Read side of the pipeline (PLAN.md section 5): per-user access scan, detect, score, embed, recognize.
Writes only to state.sqlite and the crop cache; nothing here touches Immich beyond reads."""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from tagger.config import Config
from tagger.embedder import crop
from tagger.immich import Immich, ImmichError
from tagger.matching import burst_assign, dbscan, match, split_conflicts
from tagger.quality import quality
from tagger.runtime import Models

log = logging.getLogger("scan")
CROP_THUMB_PX = 160


def sync_users(db: sqlite3.Connection, clients: dict[str, Immich]) -> dict[str, str]:
    """label -> Immich user id."""
    ids = {}
    for label, client in clients.items():
        me = client.me()
        ids[label] = me["id"]
        db.execute("INSERT INTO users(label, immich_user_id, cluster_group_id) VALUES(?, ?, ?) "
                   "ON CONFLICT(label) DO UPDATE SET immich_user_id = excluded.immich_user_id, "
                   "cluster_group_id = excluded.cluster_group_id", (label, me["id"], me.get("clusterGroupId")))
    groups = {r[0] for r in db.execute("SELECT DISTINCT cluster_group_id FROM users WHERE enabled")}
    if len(groups) > 1:
        log.warning("configured users are in %d different cluster groups; characters can't be shared until they "
                    "join one group in Immich", len(groups))
    return ids


def visible_assets(cfg: Config, client: Immich, user_id: str, album_ids: list[str] | None = None) -> dict[str, tuple[dict, str]]:
    """asset_id -> (asset, via) for one user, according to SCAN_MODE."""
    found: dict[str, tuple[dict, str]] = {}

    def add(asset: dict, via: str) -> None:
        if asset["ownerId"] == user_id:
            via = "own"
        elif via == "own":
            via = "partner"
        if via == "partner" and not cfg.include_partner:
            return
        found.setdefault(asset["id"], (asset, via))

    mode = "albums" if album_ids else cfg.scan_mode
    if mode in ("albums", "all"):
        ids = album_ids or cfg.album_ids
        if mode == "all":
            ids = [a["id"] for a in client.shared_albums()]
        for album_id in ids:
            try:
                for a in client.search_assets({"albumIds": {"any": [album_id]}}):
                    add(a, "album")
            except ImmichError as e:
                log.debug("album %s not visible: %s", album_id, e)
    if mode == "all":
        for a in client.search_assets():
            add(a, "own")
    if mode == "smart":
        for a in client.smart_search(cfg.smart_query, cfg.smart_limit):
            add(a, "own")
    return found


def taken_at(asset: dict) -> str | None:
    exif = asset.get("exifInfo") or {}
    return exif.get("dateTimeOriginal") or asset.get("fileCreatedAt")


def record_access(db: sqlite3.Connection, label: str, assets: dict[str, tuple[dict, str]]) -> None:
    db.execute("BEGIN")
    for asset_id, (a, via) in assets.items():
        db.execute("INSERT INTO assets(asset_id, owner_id, taken_at, updated_at, edited) VALUES(?, ?, ?, ?, ?) "
                   "ON CONFLICT(asset_id) DO UPDATE SET taken_at = excluded.taken_at, edited = excluded.edited, "
                   "status = CASE WHEN assets.updated_at IS NOT excluded.updated_at THEN 'new' ELSE assets.status END, "
                   "updated_at = excluded.updated_at",
                   (asset_id, a["ownerId"], taken_at(a), a.get("updatedAt"), int(bool(a.get("isEdited")))))
        db.execute("INSERT INTO asset_access(asset_id, user_label, via) VALUES(?, ?, ?) "
                   "ON CONFLICT DO UPDATE SET via = excluded.via", (asset_id, label, via))
    db.execute("COMMIT")


def pending_assets(db: sqlite3.Connection, det_v: str, emb_v: str, limit: int | None) -> list[sqlite3.Row]:
    sql = ("SELECT * FROM assets WHERE status != 'done' OR detector_version IS NOT ? OR embedder_version IS NOT ? "
           "ORDER BY taken_at")
    rows = db.execute(sql, (det_v, emb_v)).fetchall()
    return rows[:limit] if limit else rows


def reader_for(db: sqlite3.Connection, asset: sqlite3.Row, clients: dict[str, Immich], ids: dict[str, str]) -> Immich:
    """Prefer the owner's key; otherwise any configured user who can see the asset."""
    labels = [r[0] for r in db.execute("SELECT user_label FROM asset_access WHERE asset_id = ?", (asset["asset_id"],))]
    owner = [l for l in labels if ids.get(l) == asset["owner_id"]]
    return clients[(owner or labels)[0]]


def process_asset(db: sqlite3.Connection, cfg: Config, models: Models, client: Immich, asset: sqlite3.Row) -> int:
    aid = asset["asset_id"]
    img = client.preview(aid, edited=bool(asset["edited"]))
    reembed_only = asset["detector_version"] == models.detector_version and asset["status"] == "done"

    if reembed_only:
        rows = db.execute("SELECT id, x, y, w, h FROM detections WHERE asset_id = ?", (aid,)).fetchall()
        boxes = [(r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]) for r in rows]
        crops = [crop(img, b) for b in boxes]
        embs = models.embedder()([c for c in crops if c is not None])
        db.execute("BEGIN")
        for r, e in zip([r for r, c in zip(rows, crops) if c is not None], embs):
            db.execute("UPDATE detections SET embedding = ? WHERE id = ?", (e.astype(np.float32).tobytes(), r["id"]))
    else:
        boxes, scores = models.detector()(img)
        kept = [(b, s, c) for b, s in zip(boxes, scores) if (c := crop(img, b)) is not None and s >= cfg.min_score]
        embs = models.embedder()([c for _, _, c in kept]) if kept else []
        db.execute("BEGIN")
        # Rejections survive re-detection (M4 matches them by IoU); everything else is recomputed.
        db.execute("DELETE FROM detections WHERE asset_id = ? AND status != 'rejected'", (aid,))
        crops_dir = cfg.data_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for (b, s, c), e in zip(kept, embs):
            q = quality(float(s), c)
            cur = db.execute(
                "INSERT INTO detections(asset_id, x, y, w, h, img_w, img_h, score, quality, is_reference, embedding) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (aid, float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1]), img.width, img.height,
                 float(s), q, int(is_reference(cfg, float(s), q)), e.astype(np.float32).tobytes()))
            thumb = c.copy()
            thumb.thumbnail((CROP_THUMB_PX, CROP_THUMB_PX))
            thumb.save(crops_dir / f"{cur.lastrowid}.jpg", quality=85)
    db.execute("UPDATE assets SET detector_version = ?, embedder_version = ?, processed_at = datetime('now'), "
               "status = 'done' WHERE asset_id = ?", (models.detector_version, models.embedder_version, aid))
    db.execute("COMMIT")
    return len(embs)


# --- recognition ---------------------------------------------------------------------------------

def is_reference(cfg: Config, score: float, quality_: float) -> bool:
    return quality_ >= cfg.ref_quality_min and score >= cfg.ref_score_min


def _taken(db: sqlite3.Connection) -> dict[str, set[int]]:
    """asset -> characters already assigned in it (a photo can't show one character twice)."""
    taken: dict[str, set[int]] = {}
    for a, c in db.execute("SELECT asset_id, character_id FROM detections WHERE status = 'assigned'"):
        taken.setdefault(a, set()).add(c)
    return taken


def _gallery(db: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray]:
    rows = db.execute("SELECT embedding, character_id FROM detections WHERE status = 'assigned' AND is_reference "
                      "AND NOT via_burst AND embedding IS NOT NULL").fetchall()
    if not rows:
        return np.empty((0, 512), np.float32), np.empty(0, int)
    return (np.stack([np.frombuffer(r[0], np.float32) for r in rows]), np.array([r[1] for r in rows]))


def _assign_pass(db: sqlite3.Connection, cfg: Config) -> int:
    gallery, chars = _gallery(db)
    taken = _taken(db)
    assigned = 0
    pending = db.execute("SELECT id, asset_id, embedding, is_reference FROM detections WHERE status = 'pending' "
                         "AND embedding IS NOT NULL ORDER BY quality DESC").fetchall()
    for row in pending:
        e = np.frombuffer(row["embedding"], np.float32)
        m = match(e, gallery, chars, cfg.max_distance, cfg.knn)
        db.execute("UPDATE detections SET distance = ? WHERE id = ?", (None if m.distance == float("inf") else m.distance, row["id"]))
        if m.character_id is None or m.character_id in taken.get(row["asset_id"], ()):
            continue
        db.execute("UPDATE detections SET character_id = ?, status = 'assigned' WHERE id = ?", (m.character_id, row["id"]))
        taken.setdefault(row["asset_id"], set()).add(m.character_id)
        assigned += 1
        if row["is_reference"]:
            gallery, chars = np.vstack([gallery, e]), np.append(chars, m.character_id)
    return assigned


def _burst_pass(db: sqlite3.Connection, cfg: Config) -> int:
    gallery, chars = _gallery(db)
    taken = _taken(db)
    window = timedelta(minutes=cfg.burst_window_min)
    assigned = 0
    rows = db.execute("SELECT d.id, d.asset_id, d.embedding, a.taken_at FROM detections d JOIN assets a USING(asset_id) "
                      "WHERE d.status = 'pending' AND d.embedding IS NOT NULL AND a.taken_at IS NOT NULL").fetchall()
    for row in rows:
        t = datetime.fromisoformat(row["taken_at"])
        nearby = {r[0] for r in db.execute(
            "SELECT DISTINCT d.character_id FROM detections d JOIN assets a USING(asset_id) "
            "WHERE d.status = 'assigned' AND NOT d.via_burst AND d.asset_id != ? AND a.taken_at BETWEEN ? AND ?",
            (row["asset_id"], (t - window).isoformat(), (t + window).isoformat()))}
        m = match(np.frombuffer(row["embedding"], np.float32), gallery, chars, cfg.max_distance, cfg.knn)
        char = burst_assign(m, cfg.max_distance, cfg.burst_margin, nearby)
        if char is not None and char not in taken.get(row["asset_id"], ()):
            taken.setdefault(row["asset_id"], set()).add(char)
            db.execute("UPDATE detections SET character_id = ?, status = 'assigned', via_burst = 1, distance = ? "
                       "WHERE id = ?", (char, m.distance, row["id"]))
            assigned += 1
    return assigned


def _cluster_pass(db: sqlite3.Connection, cfg: Config) -> int:
    rows = db.execute("SELECT id, asset_id, embedding FROM detections WHERE status = 'pending' AND is_reference "
                      "AND embedding IS NOT NULL").fetchall()
    if len(rows) < cfg.min_faces:
        return 0
    emb = np.stack([np.frombuffer(r["embedding"], np.float32) for r in rows])
    labels = dbscan(emb, cfg.cluster_eps, cfg.min_faces)
    labels = split_conflicts(emb, np.array([r["asset_id"] for r in rows]), labels, cfg.min_faces)
    created = 0
    for label in sorted(set(labels) - {-1}):
        idx = np.flatnonzero(labels == label)
        dist = 1.0 - emb[idx] @ emb[idx].T
        np.fill_diagonal(dist, np.inf)
        char = db.execute("INSERT INTO characters DEFAULT VALUES").lastrowid
        # distance = to the nearest fellow member, so the report histogram covers founding members too
        db.executemany("UPDATE detections SET character_id = ?, status = 'assigned', distance = ? WHERE id = ?",
                       [(char, float(dist[j].min()), rows[i]["id"]) for j, i in enumerate(idx)])
        created += 1
    return created


def recognize(db: sqlite3.Connection, cfg: Config) -> dict:
    db.execute("BEGIN")
    stats = {"matched": _assign_pass(db, cfg)}
    stats["new_characters"] = _cluster_pass(db, cfg)
    if stats["new_characters"]:
        stats["matched"] += _assign_pass(db, cfg)
    stats["burst"] = _burst_pass(db, cfg)
    db.execute("COMMIT")
    return stats


def recluster(db: sqlite3.Connection, cfg: Config) -> dict:
    """Dry-run threshold tuning: forget all characters and recognize again from the stored embeddings."""
    if db.execute("SELECT 1 FROM characters WHERE person_group_id IS NOT NULL").fetchone():
        raise RuntimeError("characters are already linked to Immich people; recluster would orphan them")
    db.execute("BEGIN")
    for row in db.execute("SELECT id, score, quality FROM detections").fetchall():
        db.execute("UPDATE detections SET is_reference = ? WHERE id = ?", (int(is_reference(cfg, row[1], row[2])), row[0]))
    db.execute("UPDATE detections SET character_id = NULL, distance = NULL, via_burst = 0, status = 'pending' "
               "WHERE status = 'assigned'")
    db.execute("DELETE FROM characters")
    db.execute("COMMIT")
    return recognize(db, cfg)


def scan_once(cfg: Config, db: sqlite3.Connection, models: Models, clients: dict[str, Immich],
              album_ids: list[str] | None = None, limit: int | None = None) -> dict:
    ids = sync_users(db, clients)
    for label, client in clients.items():
        assets = visible_assets(cfg, client, ids[label], album_ids)
        record_access(db, label, assets)
        log.info("%s: %d assets in scope", label, len(assets))

    todo = pending_assets(db, models.detector_version, models.embedder_version, limit)
    log.info("%d assets to process", len(todo))
    faces = errors = 0
    for i, asset in enumerate(todo, 1):
        try:
            faces += process_asset(db, cfg, models, reader_for(db, asset, clients, ids), asset)
        except (ImmichError, OSError) as e:
            errors += 1
            if db.in_transaction:
                db.execute("ROLLBACK")
            db.execute("UPDATE assets SET status = 'error' WHERE asset_id = ?", (asset["asset_id"],))
            log.warning("asset %s failed: %s", asset["asset_id"], e)
        if i % 25 == 0:
            log.info("processed %d/%d assets, %d detections", i, len(todo), faces)
    stats = {"assets": len(todo), "detections": faces, "errors": errors, **recognize(db, cfg)}
    log.info("scan done: %s", stats)
    return stats
