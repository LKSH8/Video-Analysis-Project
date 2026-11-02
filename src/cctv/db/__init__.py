"""Database package containing repository abstraction and records."""

from cctv.db.repository import (
    DetectionRecord,
    EventRecord,
    FrameStatRecord,
    Repository,
    SQLiteRepository,
    TrackRecord,
    VideoRecord,
)

__all__ = [
    "DetectionRecord",
    "EventRecord",
    "FrameStatRecord",
    "Repository",
    "SQLiteRepository",
    "TrackRecord",
    "VideoRecord",
]
