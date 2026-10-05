"""Self-contained HTML dry-run report for threshold review (M3)."""

import base64
import html
import sqlite3
from pathlib import Path

from tagger.config import Config

CSS = """
body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
h2{margin-top:32px;border-bottom:1px solid #333;padding-bottom:4px}
.grid{display:flex;flex-wrap:wrap;gap:6px}
.c{position:relative;width:120px;font-size:11px;color:#aaa}
.c img{width:120px;height:120px;object-fit:cover;border-radius:6px;display:block}
.c.ref img{outline:2px solid #4c8}
.c.burst img{outline:2px dashed #fa3}
table{border-collapse:collapse}td,th{padding:2px 10px;text-align:right}
a{color:inherit;text-decoration:none}
"""


def _img(crops: Path, det_id: int) -> str:
    p = crops / f"{det_id}.jpg"
    return f"data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode()}" if p.exists() else ""


def _card(cfg: Config, row: sqlite3.Row) -> str:
    cls = "c" + (" burst" if row["via_burst"] else " ref" if row["is_reference"] and row["status"] == "assigned" else "")
    dist = "—" if row["distance"] is None else f"{row['distance']:.3f}"
    title = f"score {row['score']:.2f} · quality {row['quality']:.2f} · dist {dist}"
    return (f'<div class="{cls}" title="{title}"><a href="{cfg.immich_url}/photos/{row["asset_id"]}" target="_blank">'
            f'<img src="{_img(cfg.data_dir / "crops", row["id"])}"></a>q{row["quality"]:.2f} d{dist}</div>')


def build(cfg: Config, db: sqlite3.Connection) -> str:
    q = lambda sql, *a: db.execute(sql, a).fetchall()  # noqa: E731
    stats = {
        "assets processed": q("SELECT count(*) FROM assets WHERE status = 'done'")[0][0],
        "assets with errors": q("SELECT count(*) FROM assets WHERE status = 'error'")[0][0],
        "detections": q("SELECT count(*) FROM detections")[0][0],
        "assigned": q("SELECT count(*) FROM detections WHERE status = 'assigned'")[0][0],
        "  of which via burst": q("SELECT count(*) FROM detections WHERE via_burst")[0][0],
        "pending": q("SELECT count(*) FROM detections WHERE status = 'pending'")[0][0],
        "characters": q("SELECT count(*) FROM characters WHERE merged_into IS NULL")[0][0],
    }
    out = [f"<!doctype html><meta charset=utf-8><title>Fursuit tagger dry run</title><style>{CSS}</style>",
           "<h1>Fursuit tagger dry run</h1>",
           f"<p>MIN_SCORE {cfg.min_score} · MAX_DISTANCE {cfg.max_distance} · MIN_FACES {cfg.min_faces} · "
           f"REF_QUALITY_MIN {cfg.ref_quality_min} · BURST {cfg.burst_window_min} min / +{cfg.burst_margin}</p>",
           "<p>Green outline = reference crop, dashed orange = assigned via burst context. Click a crop to open the photo.</p>",
           "<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in stats.items()) + "</table>"]

    # Distance distribution helps pick MAX_DISTANCE: assigned should sit left of the line, strangers right.
    buckets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.01]
    rows = q("SELECT status, distance FROM detections WHERE distance IS NOT NULL")
    out.append("<h2>Nearest-reference distance</h2><table><tr><th>distance ≤</th><th>assigned</th><th>pending</th></tr>")
    lo = 0.0
    for hi in buckets:
        a = sum(1 for r in rows if r[0] == "assigned" and lo < r[1] <= hi)
        p = sum(1 for r in rows if r[0] == "pending" and lo < r[1] <= hi)
        out.append(f"<tr><td>{hi:.2f}</td><td>{a}</td><td>{p}</td></tr>")
        lo = hi
    out.append("</table>")

    for char in q("SELECT c.id, c.name, count(d.id) n, count(DISTINCT d.asset_id) a FROM characters c "
                  "JOIN detections d ON d.character_id = c.id AND d.status = 'assigned' "
                  "WHERE c.merged_into IS NULL GROUP BY c.id ORDER BY n DESC"):
        name = html.escape(char["name"] or f"Character #{char['id']}")
        dets = q("SELECT * FROM detections WHERE character_id = ? AND status = 'assigned' ORDER BY quality DESC", char["id"])
        out.append(f"<h2>{name} — {char['n']} crops in {char['a']} photos</h2><div class=grid>")
        out += [_card(cfg, d) for d in dets]
        out.append("</div>")

    pending = q("SELECT * FROM detections WHERE status = 'pending' ORDER BY quality DESC LIMIT 300")
    out.append(f"<h2>Pending (unassigned) — top {len(pending)} by quality</h2><div class=grid>")
    out += [_card(cfg, d) for d in pending]
    out.append("</div>")
    return "\n".join(out)
