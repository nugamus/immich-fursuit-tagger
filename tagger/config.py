"""Settings from environment variables (PLAN.md section 10). Read once at startup."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


@dataclass(frozen=True)
class Config:
    immich_url: str = field(default_factory=lambda: _env("IMMICH_URL", "http://immich-server:2283").rstrip("/"))
    users_file: Path = field(default_factory=lambda: Path(_env("USERS_FILE", "/data/users.yaml")))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "/data")))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", "/models")))

    inference_device: str = field(default_factory=lambda: _env("INFERENCE_DEVICE", "auto"))
    threads: int | None = field(default_factory=lambda: int(os.environ["THREADS"]) if os.environ.get("THREADS") else None)
    model_ttl_min: float = field(default_factory=lambda: float(_env("MODEL_TTL_MIN", "10")))

    scan_mode: str = field(default_factory=lambda: _env("SCAN_MODE", "albums"))
    album_ids: list[str] = field(default_factory=lambda: _list("ALBUM_IDS"))
    smart_query: str = field(default_factory=lambda: _env("SMART_QUERY", "fursuit"))
    smart_limit: int = field(default_factory=lambda: int(_env("SMART_LIMIT", "500")))
    include_partner: bool = field(default_factory=lambda: _env("INCLUDE_PARTNER", "false").lower() == "true")

    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0.5")))
    max_distance: float = field(default_factory=lambda: float(_env("MAX_DISTANCE", "0.15")))
    # DBSCAN eps for founding new characters from the pending pool. M3 review on 588 real photos: 0.15 missed a tight
    # 3-shot side-view group of a known suit, 0.20 formed a junk cluster of back-of-head shots; 0.17 gets the first
    # without the second. Over-split groups become extra people that the user merges in Immich.
    cluster_eps: float = field(default_factory=lambda: float(_env("CLUSTER_EPS", "0.17")))
    min_faces: int = field(default_factory=lambda: int(_env("MIN_FACES", "3")))
    ref_quality_min: float = field(default_factory=lambda: float(_env("REF_QUALITY_MIN", "0.3")))
    # Only confident heads may found a new character or act as references. M3 review: geese/ducks/human faces scored
    # <= 0.64, while blurry side views of real suits (quality 0.47-0.55, score >= 0.77) must still count.
    ref_score_min: float = field(default_factory=lambda: float(_env("REF_SCORE_MIN", "0.7")))
    knn: int = field(default_factory=lambda: int(_env("KNN", "5")))
    session_min: float = field(default_factory=lambda: float(_env("SESSION_MIN", "30")))
    burst_window_min: float = field(default_factory=lambda: float(_env("BURST_WINDOW_MIN", "10")))
    burst_margin: float = field(default_factory=lambda: float(_env("BURST_MARGIN", "0.03")))

    overlap_iou: float = field(default_factory=lambda: float(_env("OVERLAP_IOU", "0.5")))
    tag_name: str = field(default_factory=lambda: _env("TAG_NAME", "fursuit"))
    allow_untested_immich: bool = field(default_factory=lambda: _env("ALLOW_UNTESTED_IMMICH", "false").lower() == "true")
    # Full pass at least this often (picks up edits made in Immich); new photos trigger a pass sooner.
    scan_interval_min: float = field(default_factory=lambda: float(_env("SCAN_INTERVAL_MIN", "60")))
    # How often to cheaply check whether any user's photos changed (one tiny request per user).
    poll_interval_sec: float = field(default_factory=lambda: float(_env("POLL_INTERVAL_SEC", "120")))

    dry_run: bool = field(default_factory=lambda: _env("DRY_RUN", "false").lower() == "true")

    # Human people housekeeping (tagger/people.py), all opt-in.
    human_thumbnails: bool = field(default_factory=lambda: _env("HUMAN_THUMBNAILS", "false").lower() == "true")
    hide_background_pct: float = field(default_factory=lambda: float(_env("HIDE_BACKGROUND_PCT", "0")))
    share_people: bool = field(default_factory=lambda: _env("SHARE_PEOPLE", "false").lower() == "true")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.sqlite"

    def users(self) -> dict[str, str]:
        """users.yaml: `label: api-key` per line."""
        data = yaml.safe_load(self.users_file.read_text()) or {}
        if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
            raise ValueError(f"{self.users_file} must map user labels to API keys")
        return data
