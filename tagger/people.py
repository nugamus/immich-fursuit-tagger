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


def _largest_face(client: Immich, person_id: str) -> tuple[str, float, int] | None:
    """(asset_id, face width as a fraction of the image, face width in original pixels) of the person's biggest face."""
    best = None
    for asset in client.search_assets({"personIds": {"all": [person_id]}}):
        for f in client.get("/faces", id=asset["id"]):
            if (f.get("person") or {}).get("id") != person_id or not f["imageWidth"]:
                continue
            frac = (f["boundingBoxX2"] - f["boundingBoxX1"]) / f["imageWidth"]
            px = int(frac * (asset.get("width") or f["imageWidth"]))
            if best is None or px > best[2]:
                best = (asset["id"], frac, px)
    return best


def tidy(db: sqlite3.Connection, label: str, client: Immich, hide_below: float, dry_run: bool) -> dict:
    """One pass over one user's people. hide_below: hide people whose largest face is narrower than this
    fraction of the photo width (0 disables)."""
    ours = {r[0] for r in db.execute("SELECT person_group_id FROM characters WHERE person_group_id IS NOT NULL")}
    stats = {"thumbnails": 0, "hidden": 0, "skipped_user_choice": 0}
    for p in _people(client):
        if p["id"] in ours:
            continue
        best = _largest_face(client, p["id"])
        if best is None:
            continue
        asset_id, frac, _ = best
        row = db.execute("SELECT asset_id, updated_at FROM human_thumbs WHERE person_id = ? AND user_label = ?",
                         (p["id"], label)).fetchone()
        # We recorded the person's updatedAt right after our own change; a newer one means the user edited it.
        user_touched = row is not None and row[1] != p.get("updatedAt")
        if user_touched:
            stats["skipped_user_choice"] += 1
        elif row is None or row[0] != asset_id:
            if not dry_run:
                try:
                    client.put(f"/people/{p['id']}", {"featureFaceAssetId": asset_id})
                    updated = client.get(f"/people/{p['id']}")  # GET's updatedAt format, as compared later
                except ImmichError as e:
                    log.warning("thumbnail for person %s: %s", p["id"], e)
                    continue
                db.execute("INSERT INTO human_thumbs(person_id, user_label, asset_id, updated_at) VALUES(?, ?, ?, ?) "
                           "ON CONFLICT DO UPDATE SET asset_id = excluded.asset_id, updated_at = excluded.updated_at",
                           (p["id"], label, asset_id, updated.get("updatedAt")))
            stats["thumbnails"] += 1
        if hide_below and frac < hide_below and not p["isHidden"] and not p["name"] and not user_touched:
            if not dry_run:
                client.put(f"/people/{p['id']}", {"isHidden": True})
                updated = client.get(f"/people/{p['id']}")
                db.execute("UPDATE human_thumbs SET updated_at = ? WHERE person_id = ? AND user_label = ?",
                           (updated.get("updatedAt"), p["id"], label))
            stats["hidden"] += 1
    return stats


def share_all(clients: dict[str, Immich], ids: dict[str, str], dry_run: bool) -> None:
    """Share each user's people with every other configured user (write role, same cluster group required)."""
    for label, client in clients.items():
        others = [uid for other, uid in ids.items() if other != label]
        if others and not dry_run:
            try:
                client.put("/people/users", {"type": "everyone", "sharedWithIds": others, "role": "write"})
            except ImmichError as e:
                log.warning("sharing %s's people: %s", label, e)
