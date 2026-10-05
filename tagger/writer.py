"""Write side (PLAN.md section 7, Option A): one Immich person group per character, shared with every configured
user; MANUAL faces on each asset written by the asset owner's key; the fursuit tag; best thumbnails; undo.

Every ID the tagger creates is stored, so passes are idempotent and `undo` removes exactly what we made.
"""

import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np

from tagger.config import Config
from tagger.detector import iou
from tagger.immich import Immich, ImmichError
from tagger.state import get_meta, set_meta

log = logging.getLogger("writer")

TESTED_MIN = (3, 3, 0, 2)  # 3.3.0-rc.2: person sharing (PUT /people/users) and cluster groups
REQUIRED_SCOPES = {"user.read", "asset.read", "asset.view", "person.read", "person.create", "person.update",
                   "person.delete", "face.read", "face.create", "face.delete", "tag.read", "tag.create", "tag.asset",
                   "tag.delete"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- safety ----------------------------------------------------------------------------------------

def guard(cfg: Config, clients: dict[str, Immich]) -> str | None:
    """None if writing is safe, else the reason to stay read-only (PLAN.md 8.1)."""
    any_client = next(iter(clients.values()))
    major, minor, patch, pre = any_client.version()
    version = (major, minor, patch, 99 if pre is None else pre)  # a release sorts after its RCs
    if not cfg.allow_untested_immich and not (TESTED_MIN <= version and major < 4):
        return f"Immich {major}.{minor}.{patch}{'-rc.' + str(pre) if pre is not None else ''} is outside the tested range"
    groups = set()
    for label, client in clients.items():
        perms = set(client.get("/api-keys/me").get("permissions", []))
        missing = REQUIRED_SCOPES - perms if "all" not in perms else set()
        if missing:
            return f"API key for {label} lacks {sorted(missing)}"
        groups.add(client.me().get("clusterGroupId"))
    if len(groups) > 1:
        return "configured users are in different Immich cluster groups; join one group so people can be shared"
    return None


# --- helpers ---------------------------------------------------------------------------------------

def _owner_labels(db: sqlite3.Connection) -> dict[str, str]:
    """Immich user id -> label, for configured users."""
    return {r[1]: r[0] for r in db.execute("SELECT label, immich_user_id FROM users WHERE enabled")}


def _norm_box(f: dict) -> list[float]:
    w, h = f["imageWidth"] or 1, f["imageHeight"] or 1
    return [f["boundingBoxX1"] / w, f["boundingBoxY1"] / h, f["boundingBoxX2"] / w, f["boundingBoxY2"] / h]


def _det_box(d: sqlite3.Row) -> list[float]:
    return [d["x"] / d["img_w"], d["y"] / d["img_h"], (d["x"] + d["w"]) / d["img_w"], (d["y"] + d["h"]) / d["img_h"]]


class Writer:
    def __init__(self, cfg: Config, db: sqlite3.Connection, clients: dict[str, Immich]):
        self.cfg, self.db, self.clients = cfg, db, clients
        self.owners = _owner_labels(db)               # immich user id -> label
        self.ids = {v: k for k, v in self.owners.items()}  # label -> immich user id

    # --- reconcile what the user changed in Immich (minimal M4 subset; full table in M5) -------------
    def reconcile_deleted_faces(self) -> int:
        """A tagger face the user deleted in Immich becomes a permanent rejection (PLAN.md 7.1)."""
        rejected = 0
        rows = self.db.execute("SELECT id, asset_id, face_id, face_label FROM detections WHERE face_id IS NOT NULL").fetchall()
        by_asset: dict[tuple[str, str], list] = {}
        for r in rows:
            by_asset.setdefault((r["asset_id"], r["face_label"]), []).append(r)
        for (asset_id, label), dets in by_asset.items():
            try:
                present = {f["id"] for f in self.clients[label].get("/faces", id=asset_id)}
            except ImmichError as e:
                log.debug("faces of %s: %s", asset_id, e)
                continue
            for d in dets:
                if d["face_id"] not in present:
                    self.db.execute("UPDATE detections SET status = 'rejected', face_id = NULL WHERE id = ?", (d["id"],))
                    rejected += 1
        return rejected

    # --- people ------------------------------------------------------------------------------------
    def ensure_people(self) -> int:
        """Create a person group for every character that has writable faces, owned by the configured user who
        owns most of its photos, then share it with all other configured users."""
        created = 0
        chars = self.db.execute(
            "SELECT c.id, a.owner_id, count(*) n FROM characters c "
            "JOIN detections d ON d.character_id = c.id AND d.status = 'assigned' JOIN assets a USING(asset_id) "
            "WHERE c.person_group_id IS NULL AND c.merged_into IS NULL GROUP BY c.id, a.owner_id ORDER BY n DESC").fetchall()
        seen = set()
        for c in chars:
            if c["id"] in seen or c["owner_id"] not in self.owners:
                continue
            seen.add(c["id"])
            label = self.owners[c["owner_id"]]
            if self.cfg.dry_run:
                created += 1
                continue
            person = self.clients[label].post("/people", {})
            self.db.execute("UPDATE characters SET person_group_id = ?, owner_label = ?, person_created_at = ? "
                            "WHERE id = ?", (person["id"], label, now(), c["id"]))
            created += 1
            log.info("character %d -> person %s (owner %s)", c["id"], person["id"], label)
        self.share_people()
        return created

    def share_people(self) -> None:
        for c in self.db.execute("SELECT id, person_group_id, owner_label FROM characters "
                                 "WHERE person_group_id IS NOT NULL AND merged_into IS NULL").fetchall():
            others = [lab for lab in self.ids if lab != c["owner_label"]]
            missing = [lab for lab in others if not self.db.execute(
                "SELECT 1 FROM shares WHERE character_id = ? AND user_label = ?", (c["id"], lab)).fetchone()]
            if not missing or self.cfg.dry_run:
                continue
            self.clients[c["owner_label"]].put("/people/users", {
                "personIds": [c["person_group_id"]], "sharedWithIds": [self.ids[lab] for lab in missing], "role": "write"})
            self.db.executemany("INSERT OR IGNORE INTO shares(character_id, user_label) VALUES(?, ?)",
                                [(c["id"], lab) for lab in missing])

    # --- faces -------------------------------------------------------------------------------------
    def write_faces(self) -> dict:
        stats = {"written": 0, "overlap": 0, "not_owned": 0}
        rows = self.db.execute(
            "SELECT d.*, a.owner_id, c.person_group_id FROM detections d JOIN assets a USING(asset_id) "
            "JOIN characters c ON c.id = d.character_id "
            "WHERE d.status = 'assigned' AND d.face_id IS NULL AND d.skip_reason IS NULL "
            "AND (c.person_group_id IS NOT NULL OR ?) ORDER BY d.asset_id", (int(self.cfg.dry_run),)).fetchall()
        for d in rows:
            label = self.owners.get(d["owner_id"])
            if label is None:
                stats["not_owned"] += 1  # only the owner may add faces (Immich: asset.update is owner-only)
                continue
            client = self.clients[label]
            existing = client.get("/faces", id=d["asset_id"])
            box = _det_box(d)
            ours = {r[0] for r in self.db.execute("SELECT face_id FROM detections WHERE face_id IS NOT NULL")}
            clash = [f for f in existing if f["id"] not in ours and
                     iou(np.array(box), np.array([_norm_box(f)]))[0] > self.cfg.overlap_iou]
            if clash:
                # buffalo_l or the user already boxed this region; OVERLAP_POLICY=replace is reserved for M5
                self.db.execute("UPDATE detections SET skip_reason = 'overlap' WHERE id = ?", (d["id"],))
                stats["overlap"] += 1
                continue
            if self.cfg.dry_run:
                stats["written"] += 1
                continue
            before = {f["id"] for f in existing}
            client.post("/faces", {
                "assetId": d["asset_id"], "personId": d["person_group_id"],
                "imageWidth": d["img_w"], "imageHeight": d["img_h"],
                "x": round(d["x"]), "y": round(d["y"]), "width": round(d["w"]), "height": round(d["h"])})
            new = [f for f in client.get("/faces", id=d["asset_id"])
                   if f["id"] not in before and (f.get("person") or {}).get("id") == d["person_group_id"]]
            if not new:
                log.warning("face for detection %d created but not found on asset %s", d["id"], d["asset_id"])
                continue
            self.db.execute("UPDATE detections SET face_id = ?, face_label = ?, written_at = ? WHERE id = ?",
                            (new[0]["id"], label, now(), d["id"]))
            stats["written"] += 1
        return stats

    # --- tags --------------------------------------------------------------------------------------
    def _tag_id(self, label: str) -> str:
        row = self.db.execute("SELECT tag_id FROM tags WHERE user_label = ?", (label,)).fetchone()
        if row:
            return row[0]
        client = self.clients[label]
        existed = any(t.get("value") == self.cfg.tag_name for t in client.get("/tags"))
        tag = next(t for t in client.put("/tags", {"tags": [self.cfg.tag_name]}) if t.get("value") == self.cfg.tag_name)
        self.db.execute("INSERT INTO tags(user_label, tag_id, created) VALUES(?, ?, ?)", (label, tag["id"], int(not existed)))
        return tag["id"]

    def write_tags(self) -> int:
        if not self.cfg.tag_name:
            return 0
        rows = self.db.execute(
            "SELECT DISTINCT d.asset_id, a.owner_id FROM detections d JOIN assets a USING(asset_id) "
            "WHERE d.face_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM asset_tags t WHERE t.asset_id = d.asset_id)").fetchall()
        by_label: dict[str, list[str]] = {}
        for r in rows:
            if r["owner_id"] in self.owners:
                by_label.setdefault(self.owners[r["owner_id"]], []).append(r["asset_id"])
        tagged = 0
        for label, assets in by_label.items():
            if self.cfg.dry_run:
                tagged += len(assets)
                continue
            tag_id = self._tag_id(label)
            for i in range(0, len(assets), 500):
                chunk = assets[i:i + 500]
                self.clients[label].put(f"/tags/{tag_id}/assets", {"ids": chunk})
                self.db.executemany("INSERT OR IGNORE INTO asset_tags(asset_id, user_label, tag_id, written_at) "
                                    "VALUES(?, ?, ?, ?)", [(a, label, tag_id, now()) for a in chunk])
                tagged += len(chunk)
        return tagged

    # --- thumbnails --------------------------------------------------------------------------------
    def write_thumbnails(self) -> int:
        """Feature the best-quality written face each user can see, unless that user changed the person since."""
        changed = 0
        for c in self.db.execute("SELECT id, person_group_id, owner_label FROM characters "
                                 "WHERE person_group_id IS NOT NULL AND merged_into IS NULL").fetchall():
            for label in self.ids:
                best = self.db.execute(
                    "SELECT d.asset_id FROM detections d JOIN asset_access x ON x.asset_id = d.asset_id AND x.user_label = ? "
                    "WHERE d.character_id = ? AND d.face_id IS NOT NULL ORDER BY d.quality DESC, d.score DESC LIMIT 1",
                    (label, c["id"])).fetchone()
                if not best:
                    continue
                share = self.db.execute("SELECT thumb_asset_id, thumb_updated_at, thumb_name, thumbnail_locked FROM shares "
                                        "WHERE character_id = ? AND user_label = ?", (c["id"], label)).fetchone()
                if share and share["thumbnail_locked"]:
                    continue
                client = self.clients[label]
                try:
                    person = client.get(f"/people/{c['person_group_id']}")
                except ImmichError:
                    continue
                if share and share["thumb_updated_at"] and person.get("updatedAt") != share["thumb_updated_at"] \
                        and person.get("name") == (share["thumb_name"] or ""):
                    # changed by the user (not just a rename): hands off from now on
                    self.db.execute("UPDATE shares SET thumbnail_locked = 1 WHERE character_id = ? AND user_label = ?",
                                    (c["id"], label))
                    continue
                if share and share["thumb_asset_id"] == best["asset_id"]:
                    if person.get("updatedAt") != share["thumb_updated_at"]:  # renamed: remember the new state
                        self.db.execute("UPDATE shares SET thumb_updated_at = ?, thumb_name = ? WHERE character_id = ? "
                                        "AND user_label = ?", (person.get("updatedAt"), person.get("name"), c["id"], label))
                    continue
                if self.cfg.dry_run:
                    changed += 1
                    continue
                client.put(f"/people/{c['person_group_id']}", {"featureFaceAssetId": best["asset_id"]})
                # PUT and GET format updatedAt differently (ms "Z" vs us "+00:00"): store what GET returns
                updated = client.get(f"/people/{c['person_group_id']}")
                self.db.execute(
                    "INSERT INTO shares(character_id, user_label, thumb_asset_id, thumb_updated_at, thumb_name) "
                    "VALUES(?, ?, ?, ?, ?) ON CONFLICT DO UPDATE SET thumb_asset_id = excluded.thumb_asset_id, "
                    "thumb_updated_at = excluded.thumb_updated_at, thumb_name = excluded.thumb_name",
                    (c["id"], label, best["asset_id"], updated.get("updatedAt"), updated.get("name")))
                changed += 1
        return changed

    # --- one pass ----------------------------------------------------------------------------------
    def sync(self) -> dict:
        stats = {"rejected_by_user": self.reconcile_deleted_faces(), "people_created": self.ensure_people()}
        stats.update(self.write_faces())
        stats["tagged"] = self.write_tags()
        stats["thumbnails"] = self.write_thumbnails()
        log.info("write pass%s: %s", " (dry run)" if self.cfg.dry_run else "", stats)
        return stats


# --- undo ------------------------------------------------------------------------------------------

def undo(cfg: Config, db: sqlite3.Connection, clients: dict[str, Immich], since: str | None = None,
         user: str | None = None, dry_run: bool = False) -> dict:
    """Delete every face, person and tag assignment the tagger created (optionally since a date / for one user),
    then pause the loop so it doesn't immediately recreate them. Never touches anything it didn't create."""
    since = since or "0000"
    stats = {"faces": 0, "people": 0, "tagged_assets": 0, "tags": 0}
    faces = db.execute("SELECT id, face_id, face_label FROM detections WHERE face_id IS NOT NULL AND written_at >= ?"
                       + (" AND face_label = ?" if user else ""), (since, user) if user else (since,)).fetchall()
    for f in faces:
        if not dry_run:
            try:
                clients[f["face_label"]]._request("DELETE", f"/faces/{f['face_id']}", json={"force": True})
            except ImmichError as e:
                log.warning("delete face %s: %s", f["face_id"], e)
            db.execute("UPDATE detections SET face_id = NULL, face_label = NULL, written_at = NULL, skip_reason = NULL "
                       "WHERE id = ?", (f["id"],))
        stats["faces"] += 1

    tagged = db.execute("SELECT asset_id, user_label, tag_id FROM asset_tags WHERE written_at >= ?"
                        + (" AND user_label = ?" if user else ""), (since, user) if user else (since,)).fetchall()
    by_tag: dict[tuple[str, str], list[str]] = {}
    for t in tagged:
        by_tag.setdefault((t["user_label"], t["tag_id"]), []).append(t["asset_id"])
    for (label, tag_id), assets in by_tag.items():
        if not dry_run:
            clients[label]._request("DELETE", f"/tags/{tag_id}/assets", json={"ids": assets})
            db.execute("DELETE FROM asset_tags WHERE tag_id = ? AND user_label = ? AND written_at >= ?", (tag_id, label, since))
        stats["tagged_assets"] += len(assets)

    people = db.execute("SELECT id, person_group_id, owner_label FROM characters WHERE person_group_id IS NOT NULL "
                        "AND person_created_at >= ?" + (" AND owner_label = ?" if user else ""),
                        (since, user) if user else (since,)).fetchall()
    for p in people:
        if not dry_run:
            try:
                clients[p["owner_label"]]._request("DELETE", f"/people/{p['person_group_id']}")
            except ImmichError as e:
                log.warning("delete person %s: %s", p["person_group_id"], e)
            db.execute("UPDATE characters SET person_group_id = NULL, owner_label = NULL, person_created_at = NULL "
                       "WHERE id = ?", (p["id"],))
            db.execute("DELETE FROM shares WHERE character_id = ?", (p["id"],))
        stats["people"] += 1

    if since == "0000" and not user:
        for t in db.execute("SELECT user_label, tag_id FROM tags WHERE created").fetchall():
            if not dry_run:
                try:
                    clients[t["user_label"]]._request("DELETE", f"/tags/{t['tag_id']}")
                except ImmichError as e:
                    log.warning("delete tag %s: %s", t["tag_id"], e)
                db.execute("DELETE FROM tags WHERE tag_id = ?", (t["tag_id"],))
            stats["tags"] += 1
    if not dry_run:
        set_meta(db, "paused", "1")
    log.info("undo%s: %s", " (dry run)" if dry_run else "", stats)
    return stats


def paused(db: sqlite3.Connection) -> bool:
    return get_meta(db, "paused") == "1"
