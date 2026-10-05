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
    min_faces: int = field(default_factory=lambda: int(_env("MIN_FACES", "3")))
    ref_quality_min: float = field(default_factory=lambda: float(_env("REF_QUALITY_MIN", "0.5")))
    knn: int = field(default_factory=lambda: int(_env("KNN", "5")))
    burst_window_min: float = field(default_factory=lambda: float(_env("BURST_WINDOW_MIN", "10")))
    burst_margin: float = field(default_factory=lambda: float(_env("BURST_MARGIN", "0.05")))

    dry_run: bool = field(default_factory=lambda: _env("DRY_RUN", "false").lower() == "true")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.sqlite"

    def users(self) -> dict[str, str]:
        """users.yaml: `label: api-key` per line."""
        data = yaml.safe_load(self.users_file.read_text()) or {}
        if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
            raise ValueError(f"{self.users_file} must map user labels to API keys")
        return data
