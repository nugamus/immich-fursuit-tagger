import numpy as np

from tagger import state
from tagger.scan import _copy_pass


def _emb(*v):
    a = np.array(v + (0,) * (512 - len(v)), dtype=np.float32)
    return (a / np.linalg.norm(a)).tobytes()


def _asset(db, asset_id, stem, taken):
    db.execute("INSERT INTO assets(asset_id, owner_id, taken_at, file_stem) VALUES(?, 'o', ?, ?)", (asset_id, taken, stem))


def _det(db, asset_id, emb, status="pending", char=None):
    return db.execute("INSERT INTO detections(asset_id, x, y, w, h, img_w, img_h, score, quality, embedding, status, "
                      "character_id) VALUES(?, 0, 0, 1, 1, 1, 1, 0.9, 0.9, ?, ?, ?)",
                      (asset_id, emb, status, char)).lastrowid


def test_edited_copy_inherits_characters_and_capture_time(tmp_path):
    db = state.connect(tmp_path / "s.sqlite")
    db.execute("INSERT INTO characters(id) VALUES (5), (6)")
    _asset(db, "raw", "z61_1659", "2026-08-28T06:16:25+00:00")
    _asset(db, "jpg", "z61_1659", "2026-10-03T14:30:00+00:00")  # export with a placeholder date
    _det(db, "raw", _emb(1, 0), "assigned", 5)
    _det(db, "raw", _emb(0, 1), "rejected", 6)
    a = _det(db, "jpg", _emb(1, 0.05))
    b = _det(db, "jpg", _emb(0.05, 1))

    assert _copy_pass(db) == 2
    got = {r[0]: (r[1], r[2]) for r in db.execute("SELECT id, status, character_id FROM detections")}
    assert got[a] == ("assigned", 5) and got[b] == ("rejected", 6)
    assert tuple(db.execute("SELECT copy_of, taken_at FROM assets WHERE asset_id = 'jpg'").fetchone()) == \
        ("raw", "2026-08-28T06:16:25+00:00")
    assert _copy_pass(db) == 0  # stable: the copy stays the copy now that both share a date


def test_same_file_name_but_different_photo_is_not_linked(tmp_path):
    db = state.connect(tmp_path / "s.sqlite")
    db.execute("INSERT INTO characters(id) VALUES (5)")
    _asset(db, "mine", "dsc_1234", "2026-08-28T06:00:00+00:00")
    _asset(db, "theirs", "dsc_1234", "2026-09-01T12:00:00+00:00")  # another photographer's camera counter
    _det(db, "mine", _emb(1, 0), "assigned", 5)
    other = _det(db, "theirs", _emb(0.6, 1))

    assert _copy_pass(db) == 0
    row = db.execute("SELECT status, character_id FROM detections WHERE id = ?", (other,)).fetchone()
    assert tuple(row) == ("pending", None)
    assert db.execute("SELECT copy_of FROM assets WHERE asset_id = 'theirs'").fetchone()[0] is None
