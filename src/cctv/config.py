"""Configuration loader and schema validation using Pydantic v2."""

from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    camera_id: str = "cam1"
    db_path: str = "data/cctv.db"
    evidence_dir: str = "data/evidence"
    videos_dir: str = "data/videos"


class IngestConfig(BaseModel):
    frame_stride: int = Field(default=4, ge=1)


class PerceiveConfig(BaseModel):
    model_name: str = "yolo11n.pt"
    tracker_config: str = "bytetrack.yaml"
    imgsz: int = Field(default=640, ge=160, le=1280)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    min_track_seconds: float = Field(default=0.5, ge=0.0)
    target_classes: List[str] = [
        "person",
        "car",
        "laptop",
        "backpack",
        "handbag",
        "suitcase",
        "cell phone",
        "bottle",
        "book",
        "cup",
        "chair",
    ]


class EventsConfig(BaseModel):
    object_stable_seconds: float = Field(default=1.0, ge=0.0)
    object_missing_seconds: float = Field(default=3.0, ge=0.0)
    object_lookback_seconds: float = Field(default=2.0, ge=0.0)
    near_ratio: float = Field(default=1.0, ge=0.0)
    near_min_seconds: float = Field(default=1.0, ge=0.0)
    loiter_radius_ratio: float = Field(default=0.5, ge=0.0)
    loiter_seconds: float = Field(default=10.0, ge=0.0)
    near_px_ratio: float = Field(default=1.0, ge=0.0)
    occlusion_iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class EvidenceConfig(BaseModel):
    pre_roll_seconds: float = Field(default=2.0, ge=0.0)
    post_roll_seconds: float = Field(default=2.0, ge=0.0)
    max_clip_seconds: float = Field(default=20.0, ge=1.0)


class QueryConfig(BaseModel):
    parser: str = "hybrid"  # "rules", "gemini", "hybrid"
    default_limit: int = Field(default=20, ge=1)
    temporal_window_seconds: float = Field(default=30.0, ge=0.0)


class LLMConfig(BaseModel):
    model: str = "gemini-3.5-flash-lite"
    timeout_seconds: float = Field(default=10.0, ge=1.0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    perceive: PerceiveConfig = Field(default_factory=PerceiveConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


_CONFIG_CACHE: Optional[AppConfig] = None


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        # Fallback to default config if file is missing
        return AppConfig()
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    global _CONFIG_CACHE
    _CONFIG_CACHE = AppConfig.model_validate(data)
    return _CONFIG_CACHE


def get_config() -> AppConfig:
    """Retrieve the cached application config or load default."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE
