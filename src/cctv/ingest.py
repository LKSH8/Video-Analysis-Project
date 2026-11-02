"""Video ingest module: reads metadata and provides timestamped frame iterator."""

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float
    start_wallclock: Optional[str] = None


def read_video_metadata(
    video_path: str | Path, start_time: Optional[str] = None
) -> VideoMetadata:
    """Inspect video file and return its properties.
    
    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV cannot open or read the video.
    """
    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file does not exist: {path_obj}")

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video file: {path_obj}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
            # Try reading at least one frame if frame count/fps is weird
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Cannot read valid frames from video: {path_obj}")
            if height <= 0 or width <= 0:
                height, width = frame.shape[:2]
            if fps <= 0:
                fps = 30.0
            if frame_count <= 0:
                frame_count = 1

        duration_s = float(frame_count) / fps

        # Validate start_wallclock if provided
        validated_wallclock = None
        if start_time:
            try:
                dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                validated_wallclock = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                logger.warning(
                    "Invalid start_time format '%s'. Expected 'YYYY-MM-DD HH:MM:SS'. Setting to None.",
                    start_time,
                )

        return VideoMetadata(
            path=str(path_obj.as_posix()),
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_s=duration_s,
            start_wallclock=validated_wallclock,
        )
    finally:
        cap.release()


def frame_iterator(
    video_path: str | Path,
    frame_stride: int = 1,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Yields (frame_index, t_seconds, frame) for each frame in the video.
    
    Timestamps are precisely calculated as frame_index / fps.
    """
    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file does not exist: {path_obj}")

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path_obj}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    stride = max(1, frame_stride)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % stride == 0:
                t_seconds = float(frame_idx) / fps
                yield (frame_idx, t_seconds, frame)

            frame_idx += 1
    finally:
        cap.release()
