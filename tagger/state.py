"""SQLite state (PLAN.md section 9). WAL mode; migrations keyed on PRAGMA user_version."""

import sqlite3
from pathlib import Path

MIGRATIONS = [
    """
    CREATE TABLE users (
        label TEXT PRIMARY KEY,
        immich_user_id TEXT NOT NULL,
        cluster_group_id TEXT,
        enabled INTEGER NOT NULL DEFAULT 1
    );
    CREATE TABLE assets (
        asset_id TEXT PRIMARY KEY,
        owner_id TEXT NOT NULL,
        taken_at TEXT,
        updated_at TEXT,
        edited INTEGER NOT NULL DEFAULT 0,
        detector_version TEXT,
        embedder_version TEXT,
        processed_at TEXT,
        status TEXT NOT NULL DEFAULT 'new'          -- new | done | error
    );
    CREATE INDEX assets_taken_at ON assets(taken_at);
    CREATE TABLE asset_access (
        asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
        user_label TEXT NOT NULL,
        via TEXT NOT NULL,                          -- own | album | partner
        PRIMARY KEY (asset_id, user_label)
    );
    CREATE TABLE characters (
        id INTEGER PRIMARY KEY,
        person_group_id TEXT UNIQUE,
        name TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        merged_into INTEGER REFERENCES characters(id)
    );
    CREATE TABLE detections (
        id INTEGER PRIMARY KEY,
        asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
        x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
        img_w INTEGER NOT NULL, img_h INTEGER NOT NULL,
        score REAL NOT NULL,
        quality REAL NOT NULL,
        is_reference INTEGER NOT NULL DEFAULT 0,
        via_burst INTEGER NOT NULL DEFAULT 0,
        embedding BLOB,
        character_id INTEGER REFERENCES characters(id),
        distance REAL,
        status TEXT NOT NULL DEFAULT 'pending',     -- pending | assigned | rejected
        face_id TEXT
    );
    CREATE INDEX detections_asset ON detections(asset_id);
    CREATE INDEX detections_character ON detections(character_id);
    CREATE TABLE shares (
        character_id INTEGER NOT NULL REFERENCES characters(id),
        user_label TEXT NOT NULL,
        thumbnail_locked INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (character_id, user_label)
    );
    CREATE TABLE asset_tags (
        asset_id TEXT NOT NULL,
        user_label TEXT NOT NULL,
        tag_id TEXT NOT NULL,
        PRIMARY KEY (asset_id, user_label)
    );
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """,
    """
    CREATE TABLE human_thumbs (
        person_id TEXT NOT NULL,
        user_label TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        updated_at TEXT,                            -- person.updatedAt right after our change
        PRIMARY KEY (person_id, user_label)
    );
    """,
    """
    ALTER TABLE characters ADD COLUMN owner_label TEXT;
    ALTER TABLE characters ADD COLUMN person_created_at TEXT;
    ALTER TABLE detections ADD COLUMN face_label TEXT;
    ALTER TABLE detections ADD COLUMN written_at TEXT;
    ALTER TABLE detections ADD COLUMN skip_reason TEXT;
    ALTER TABLE shares ADD COLUMN thumb_asset_id TEXT;
    ALTER TABLE shares ADD COLUMN thumb_updated_at TEXT;
    ALTER TABLE shares ADD COLUMN thumb_name TEXT;
    ALTER TABLE asset_tags ADD COLUMN written_at TEXT;
    CREATE TABLE tags (user_label TEXT PRIMARY KEY, tag_id TEXT NOT NULL, created INTEGER NOT NULL DEFAULT 0);
    """,
    """
    CREATE TABLE person_shares (character_id INTEGER NOT NULL, user_id TEXT NOT NULL, PRIMARY KEY (character_id, user_id));
    """,
    """
    ALTER TABLE human_thumbs ADD COLUMN n_assets INTEGER;
    ALTER TABLE human_thumbs ADD COLUMN largest REAL;
    ALTER TABLE human_thumbs ADD COLUMN name TEXT;
    ALTER TABLE human_thumbs ADD COLUMN locked INTEGER NOT NULL DEFAULT 0;
    """,
]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=5000")
    version = db.execute("PRAGMA user_version").fetchone()[0]
    for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
        db.execute("BEGIN")
        _run_script(db, sql)
        db.execute(f"PRAGMA user_version={i}")
        db.execute("COMMIT")
    return db


def _run_script(db: sqlite3.Connection, sql: str) -> None:
    # executescript() would COMMIT the open transaction; run statements one by one instead.
    for statement in sql.split(";"):
        if statement.strip():
            db.execute(statement)


def get_meta(db: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
               (key, value))
