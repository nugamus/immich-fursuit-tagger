import numpy as np

from tagger import state
from tagger.config import Config
from tagger.scan import _session_pass


def _emb(*v):
    a = np.array(v + (0,) * (512 - len(v)), dtype=np.float32)
    return (a / np.linalg.norm(a)).tobytes()


def test_session_pass_chains_within_session_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_DISTANCE", "0.15")
    monkeypatch.setenv("SESSION_MIN", "30")
    cfg = Config()
    db = state.connect(tmp_path / "s.sqlite")
    db.execute("INSERT INTO characters(id) VALUES (6)")
    # a0 is assigned; a1 is close to a0, a2 close to a1 only (chain); a3 close to a0 but 2 hours later.
    for aid, t in [("a0", "10:00"), ("a1", "10:01"), ("a2", "10:02"), ("a3", "12:30")]:
        db.execute("INSERT INTO assets(asset_id, owner_id, taken_at) VALUES(?, 'o', ?)", (aid, f"2026-08-28T{t}:00+00:00"))
    base = (1.0, 0.0)
    rows = [("a0", _emb(*base), "assigned", 6), ("a1", _emb(1.0, 0.45), "pending", None),
            ("a2", _emb(1.0, 0.9), "pending", None), ("a3", _emb(1.0, 0.1), "pending", None)]
    for aid, e, st, c in rows:
        db.execute("INSERT INTO detections(asset_id, x, y, w, h, img_w, img_h, score, quality, embedding, status, "
                   "character_id) VALUES(?, 0, 0, 1, 1, 1, 1, 0.9, 0.9, ?, ?, ?)", (aid, e, st, c))
    assert _session_pass(db, cfg) == 2
    got = dict(db.execute("SELECT asset_id, character_id FROM detections").fetchall())
    assert got == {"a0": 6, "a1": 6, "a2": 6, "a3": None}
