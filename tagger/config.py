"""Settings, read from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _float(name: str, default: str) -> float:
    return float(_env(name, default))


def _bool(name: str, default: str = "false") -> bool:
    return _env(name, default).strip().lower() in ("1", "true", "yes")


def _list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def parse_keys(text: str) -> dict[str, str]:
    """`name=key` pairs separated by commas or newlines."""
    keys = {}
    for item in text.replace("\n", ",").split(","):
        if item.strip():
            name, sep, key = item.partition("=")
            if not sep or not name.strip() or not key.strip():
                raise ValueError(f"expected name=key, got {item.strip()[:20]!r}")
            keys[name.strip()] = key.strip()
    return keys


@dataclass(frozen=True)
class Config:
    immich_url: str = field(default_factory=lambda: _env("IMMICH_URL", "http://immich-server:2283").rstrip("/"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "/data")))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", "/models")))

    # Hardware
    inference_device: str = field(default_factory=lambda: _env("INFERENCE_DEVICE", "auto"))
    threads: int | None = field(default_factory=lambda: int(os.environ["THREADS"]) if os.environ.get("THREADS") else None)
    model_ttl_min: float = field(default_factory=lambda: _float("MODEL_TTL_MIN", "10"))

    # What to scan
    scan_mode: str = field(default_factory=lambda: _env("SCAN_MODE", "all"))
    album_ids: list[str] = field(default_factory=lambda: _list("ALBUM_IDS"))
    smart_query: str = field(default_factory=lambda: _env("SMART_QUERY", "fursuit"))
    smart_limit: int = field(default_factory=lambda: int(_env("SMART_LIMIT", "500")))
    include_partner: bool = field(default_factory=lambda: _bool("INCLUDE_PARTNER"))

    # Recognition. Defaults were tuned on a real 600-photo library; see the README for what each one trades off.
    min_score: float = field(default_factory=lambda: _float("MIN_SCORE", "0.5"))
    max_distance: float = field(default_factory=lambda: _float("MAX_DISTANCE", "0.15"))
    cluster_eps: float = field(default_factory=lambda: _float("CLUSTER_EPS", "0.17"))
    min_faces: int = field(default_factory=lambda: int(_env("MIN_FACES", "3")))
    ref_quality_min: float = field(default_factory=lambda: _float("REF_QUALITY_MIN", "0.3"))
    # Animals and human faces that fool the detector score low; they may never start a character.
    ref_score_min: float = field(default_factory=lambda: _float("REF_SCORE_MIN", "0.7"))
    knn: int = field(default_factory=lambda: int(_env("KNN", "5")))
    session_min: float = field(default_factory=lambda: _float("SESSION_MIN", "30"))
    burst_window_min: float = field(default_factory=lambda: _float("BURST_WINDOW_MIN", "10"))
    burst_margin: float = field(default_factory=lambda: _float("BURST_MARGIN", "0.03"))

    # Writing to Immich
    dry_run: bool = field(default_factory=lambda: _bool("DRY_RUN"))
    overlap_iou: float = field(default_factory=lambda: _float("OVERLAP_IOU", "0.5"))
    tag_name: str = field(default_factory=lambda: _env("TAG_NAME", "fursuit"))
    allow_untested_immich: bool = field(default_factory=lambda: _bool("ALLOW_UNTESTED_IMMICH"))

    # Schedule: a full pass at least every SCAN_INTERVAL_MIN; new uploads are noticed within POLL_INTERVAL_SEC.
    scan_interval_min: float = field(default_factory=lambda: _float("SCAN_INTERVAL_MIN", "60"))
    poll_interval_sec: float = field(default_factory=lambda: _float("POLL_INTERVAL_SEC", "120"))

    # Housekeeping for Immich's own (human) people
    share_people: bool = field(default_factory=lambda: _bool("SHARE_PEOPLE", "true"))
    human_thumbnails: bool = field(default_factory=lambda: _bool("HUMAN_THUMBNAILS", "true"))
    hide_background_pct: float = field(default_factory=lambda: _float("HIDE_BACKGROUND_PCT", "0"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.sqlite"

    def users(self) -> dict[str, str]:
        """API keys per user, from API_KEYS or the file named by API_KEYS_FILE."""
        if os.environ.get("API_KEYS"):
            keys = parse_keys(os.environ["API_KEYS"])
        elif os.environ.get("API_KEYS_FILE"):
            keys = parse_keys(Path(os.environ["API_KEYS_FILE"]).read_text())
        else:
            keys = {}
        if not keys:
            raise ValueError("set API_KEYS to name=key pairs, e.g. API_KEYS=alice=xxxx,bob=yyyy")
        return keys
