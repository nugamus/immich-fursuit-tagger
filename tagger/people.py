"""Housekeeping for Immich's own (human) people, opt-in. Never touches fursuit characters the tagger owns.

- Best thumbnail: feature each person's largest face (biggest = least blurry), unless the user picked one themselves.
- Background people: optionally hide people whose largest face stays tiny (hidden, never deleted; unhide in Immich).
- Sharing: keep every configured user's people shared with the others (Immich's "everyone" share is a snapshot,
  so it is re-applied each pass to cover people created since).
"""

import logging
import sqlite3

from tagger.immich import Immich, ImmichError

log = logging.getLogger("people")


def _people(client: Immich) -> list[dict]:
    out, page = [], 1
    while True:
        r = client.get("/people", withHidden="true", page=page, size=500)
        out += r["people"]
        if not r.get("hasNextPage"):
            return out
        page += 1


def _face_crop(client: Immich, asset_id: str, f: dict):
    img = client.preview(asset_id, edited=True)
    sx, sy = img.width / f["imageWidth"], img.height / f["imageHeight"]
    return img.crop((int(f["boundingBoxX1"] * sx), int(f["boundingBoxY1"] * sy),
                     int(f["boundingBoxX2"] * sx), int(f["boundingBoxY2"] * sy)))


def face_score(crop) -> float:
    """Bigger, well-exposed, sharp faces win. Size alone picked a nearly black frame (M4 review)."""
    from PIL import ImageStat

    from tagger.quality import sharpness

    brightness = ImageStat.Stat(crop.convert("L")).mean[0]
    exposure = max(0.05, min(1.0, brightness / 90.0)) * max(0.05, min(1.0, (255 - brightness) / 60.0))
    s = sharpness(crop)
    return min(crop.width, 400) * exposure * (s / (s + 100.0))


def _best_face(client: Immich, person_id: str) -> tuple[str, float, int] | None:
    """(asset_id, largest face width as a fraction of the photo, number of photos) for the best-looking face."""
    best, largest, n = None, 0.0, 0
    for asset in client.search_assets({"personIds": {"all": [person_id]}}):
        n += 1
        for f in client.get("/faces", id=asset["id"]):
            if (f.get("person") or {}).get("id") != person_id or not f["imageWidth"]:
                continue
            largest = max(largest, (f["boundingBoxX2"] - f["boundingBoxX1"]) / f["imageWidth"])
            try:
                score = face_score(_face_crop(client, asset["id"], f))
            except (ImmichError, OSError):
                continue
            if best is None or score > best[1]:
                best = (asset["id"], score)
    return (best[0], largest, n) if best else None


def tidy(db: sqlite3.Connection, label: str, client: Immich, hide_below: float, dry_run: bool) -> dict:
    """One pass over one user's people. hide_below: hide people whose largest face is narrower than this
    fraction of the photo width (0 disables)."""
    ours = {r[0] for r in db.execute("SELECT person_group_id FROM characters WHERE person_group_id IS NOT NULL")}
    stats = {"thumbnails": 0, "hidden": 0, "skipped_user_choice": 0}
    for p in _people(client):
        if p["id"] in ours:
            continue
        row = db.execute("SELECT asset_id, updated_at, name, n_assets, largest, locked FROM human_thumbs "
                         "WHERE person_id = ? AND user_label = ?", (p["id"], label)).fetchone()
        if row is not None and row["locked"]:
            stats["skipped_user_choice"] += 1
            continue
        # We store the person's updatedAt right after each of our own changes. A newer one with the same name means
        # the user picked a thumbnail (hands off for good); a newer one with a new name is just a rename.
        if row is not None and row["updated_at"] != p.get("updatedAt") and row["name"] == p["name"]:
            db.execute("UPDATE human_thumbs SET locked = 1 WHERE person_id = ? AND user_label = ?", (p["id"], label))
            stats["skipped_user_choice"] += 1
            continue
        n_now = sum(1 for _ in client.search_assets({"personIds": {"all": [p["id"]]}}))
        if row is not None and row["n_assets"] == n_now:
            asset_id, frac = row["asset_id"], row["largest"] or 0  # nothing new: skip the per-face scoring
        else:
            best = _best_face(client, p["id"])
            if best is None:
                continue
            asset_id, frac, _ = best
        if dry_run:
            stats["thumbnails"] += int(row is None or row["asset_id"] != asset_id)
            continue
        try:
            if row is None or row["asset_id"] != asset_id:
                client.put(f"/people/{p['id']}", {"featureFaceAssetId": asset_id})
                stats["thumbnails"] += 1
            if hide_below and frac < hide_below and not p["isHidden"] and not p["name"]:
                client.put(f"/people/{p['id']}", {"isHidden": True})
                stats["hidden"] += 1
            current = client.get(f"/people/{p['id']}")  # GET's updatedAt format, as compared next pass
        except ImmichError as e:
            log.warning("person %s: %s", p["id"], e)
            continue
        db.execute("INSERT INTO human_thumbs(person_id, user_label, asset_id, updated_at, name, n_assets, largest) "
                   "VALUES(?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO UPDATE SET asset_id = excluded.asset_id, "
                   "updated_at = excluded.updated_at, name = excluded.name, n_assets = excluded.n_assets, "
                   "largest = excluded.largest",
                   (p["id"], label, asset_id, current.get("updatedAt"), current.get("name"), n_now, frac))
    return stats


def share_all(clients: dict[str, Immich], ids: dict[str, str], dry_run: bool) -> None:
    """Share each user's people with every other configured user (write role, same cluster group required)."""
    for label, client in clients.items():
        group = client.cluster_members() or []
        others = sorted(({uid for uid in ids.values()} | set(group)) - {ids[label]})
        if others and not dry_run:
            try:
                client.put("/people/users", {"type": "everyone", "sharedWithIds": others, "role": "write"})
            except ImmichError as e:
                log.warning("sharing %s's people: %s", label, e)
