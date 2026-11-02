"""Perception module: YOLO11n + ByteTrack detection, tracking, and flicker filtering."""

from collections import defaultdict
from dataclasses import dataclass
import logging
import math
from typing import Callable, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from cctv.config import PerceiveConfig, get_config
from cctv.db.repository import DetectionRecord, FrameStatRecord, TrackRecord
from cctv.ingest import VideoMetadata, frame_iterator

logger = logging.getLogger(__name__)


@dataclass
class RawDetection:
    frame: int
    t: float
    track_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def extract_dominant_color(crop: Optional[np.ndarray]) -> str:
    """Extract dominant visual color (red, blue, green, yellow, white, black, gray, orange, purple)."""
    if crop is None or crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
        return "unknown"

    h, w = crop.shape[:2]
    # Sample the inner 70% to exclude background surrounding the object
    center = crop[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    if center.size == 0:
        center = crop

    small = cv2.resize(center, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Threshold masks
    black_mask = V < 45
    white_mask = (S < 35) & (V > 185)
    gray_mask = (S < 40) & (V >= 45) & (V <= 185)

    chroma_mask = ~(black_mask | white_mask | gray_mask)
    H_chroma = H[chroma_mask]

    counts = {
        "black": int(np.sum(black_mask)),
        "white": int(np.sum(white_mask)),
        "gray": int(np.sum(gray_mask)),
        "red": int(np.sum((H_chroma < 10) | (H_chroma >= 165))),
        "orange": int(np.sum((H_chroma >= 10) & (H_chroma < 25))),
        "yellow": int(np.sum((H_chroma >= 25) & (H_chroma < 35))),
        "green": int(np.sum((H_chroma >= 35) & (H_chroma < 85))),
        "blue": int(np.sum((H_chroma >= 85) & (H_chroma < 130))),
        "purple": int(np.sum((H_chroma >= 130) & (H_chroma < 165))),
    }
    return max(counts.items(), key=lambda x: x[1])[0]


def run_perception(
    video_path: str,
    metadata: VideoMetadata,
    config: Optional[PerceiveConfig] = None,
    frame_stride: int = 1,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[List[TrackRecord], List[DetectionRecord], List[FrameStatRecord]]:
    """Runs object detection & ByteTrack on the video frames.
    
    Filters out flickering tracks that last less than min_track_seconds.
    
    Args:
        video_path: Path to the video file.
        metadata: Metadata of the video.
        config: PerceiveConfig (defaults to system config if None).
        frame_stride: Process every N-th frame.
        progress_callback: Optional callback(current_frame, total_frames, percentage).

    Returns:
        (tracks, detections, frame_stats)
    """
    if config is None:
        config = get_config().perceive

    logger.info("Initializing YOLO model: %s", config.model_name)
    model = YOLO(config.model_name)

    # Map target class names to class IDs if available in model.names
    allowed_classes: Set[str] = set(config.target_classes)
    model_names: Dict[int, str] = model.names

    raw_detections: List[RawDetection] = []
    # Track metadata accumulator: track_id -> dict
    track_acc: Dict[int, Dict] = {}

    total_frames = metadata.frame_count
    stride = max(1, frame_stride)

    # Stats aggregation per whole second
    second_people: Dict[int, int] = {}
    second_objects: Dict[int, int] = {}

    logger.info("Starting frame tracking with ByteTrack (%s)...", config.tracker_config)

    for frame_idx, t_sec, frame in frame_iterator(video_path, frame_stride=stride):
        # Update progress
        if progress_callback:
            pct = min(1.0, float(frame_idx + 1) / max(1, total_frames))
            progress_callback(frame_idx + 1, total_frames, pct)

        # Run model tracking
        results = model.track(
            source=frame,
            persist=True,
            tracker=config.tracker_config,
            conf=config.confidence_threshold,
            imgsz=config.imgsz,
            verbose=False,
        )

        current_frame_people = 0
        current_frame_objects = 0

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # ByteTrack assigns box.id when tracking is active
                if box.id is None:
                    continue

                track_id = int(box.id[0].item())
                cls_id = int(box.cls[0].item())
                class_name = model_names.get(cls_id, "unknown")
                conf = float(box.conf[0].item())

                # Filter target classes
                if class_name not in allowed_classes:
                    continue

                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                if class_name == "person":
                    current_frame_people += 1
                else:
                    current_frame_objects += 1

                raw_detections.append(
                    RawDetection(
                        frame=frame_idx,
                        t=t_sec,
                        track_id=track_id,
                        class_name=class_name,
                        conf=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )

                crop = frame[max(0, int(y1)):min(frame.shape[0], int(y2)), max(0, int(x1)):min(frame.shape[1], int(x2))]

                if track_id not in track_acc:
                    track_acc[track_id] = {
                        "class": class_name,
                        "first_t": t_sec,
                        "last_t": t_sec,
                        "n_frames": 1,
                        "best_conf": conf,
                        "best_frame_t": t_sec,
                        "best_crop": crop.copy() if crop.size > 0 else None,
                    }
                else:
                    acc = track_acc[track_id]
                    acc["last_t"] = t_sec
                    acc["n_frames"] += 1
                    if conf > acc["best_conf"]:
                        acc["best_conf"] = conf
                        acc["best_frame_t"] = t_sec
                        if crop.size > 0:
                            acc["best_crop"] = crop.copy()

        # Record max counts for this integer second
        sec_int = int(t_sec)
        second_people[sec_int] = max(second_people.get(sec_int, 0), current_frame_people)
        second_objects[sec_int] = max(second_objects.get(sec_int, 0), current_frame_objects)

    # Filter flicker: tracks shorter than min_track_seconds
    valid_track_ids: Set[int] = set()
    initial_track_records: List[TrackRecord] = []

    for track_id, acc in track_acc.items():
        duration = acc["last_t"] - acc["first_t"]
        if duration >= config.min_track_seconds:
            valid_track_ids.add(track_id)
            color = extract_dominant_color(acc.get("best_crop"))
            initial_track_records.append(
                TrackRecord(
                    video_id=0,
                    track_id=track_id,
                    class_name=acc["class"],
                    first_t=round(acc["first_t"], 3),
                    last_t=round(acc["last_t"], 3),
                    n_frames=acc["n_frames"],
                    best_conf=round(acc["best_conf"], 3),
                    best_frame_t=round(acc["best_frame_t"], 3),
                    color=color,
                )
            )

    # De-fragment tracks: merge split tracks of the same class across occlusions
    track_records, raw_detections = _merge_fragmented_tracks(
        tracks=initial_track_records,
        detections=raw_detections,
        max_gap_s=3.0,
    )
    final_valid_ids = {t.track_id for t in track_records}

    logger.info(
        "Perception complete. Found %d raw tracks -> %d after flicker filter -> %d after track de-fragmentation.",
        len(track_acc),
        len(initial_track_records),
        len(track_records),
    )

    # Filter detections to only include final valid tracks
    detection_records: List[DetectionRecord] = [
        DetectionRecord(
            video_id=0,
            frame=d.frame,
            t=round(d.t, 3),
            track_id=d.track_id,
            class_name=d.class_name,
            conf=round(d.conf, 3),
            x1=round(d.x1, 1),
            y1=round(d.y1, 1),
            x2=round(d.x2, 1),
            y2=round(d.y2, 1),
        )
        for d in raw_detections
        if d.track_id in final_valid_ids
    ]

    # Build FrameStatRecords (one row per second from 0 to video duration)
    total_seconds = int(np.ceil(metadata.duration_s))
    frame_stat_records: List[FrameStatRecord] = []
    for s in range(total_seconds + 1):
        frame_stat_records.append(
            FrameStatRecord(
                video_id=0,
                t=float(s),
                n_people=second_people.get(s, 0),
                n_objects=second_objects.get(s, 0),
            )
        )

    return track_records, detection_records, frame_stat_records


def _merge_fragmented_tracks(
    tracks: List[TrackRecord],
    detections: List[RawDetection],
    max_gap_s: float = 3.0,
) -> Tuple[List[TrackRecord], List[RawDetection]]:
    """Merges split tracks of the same class that broke apart due to momentary occlusion."""
    if not tracks:
        return tracks, detections

    dets_by_track: Dict[int, List[RawDetection]] = defaultdict(list)
    for d in detections:
        dets_by_track[d.track_id].append(d)

    for tid in dets_by_track:
        dets_by_track[tid].sort(key=lambda d: d.t)

    sorted_tracks = sorted(tracks, key=lambda t: t.first_t)
    id_mapping: Dict[int, int] = {}
    merged_tracks: List[TrackRecord] = []

    for curr in sorted_tracks:
        merged = False
        curr_dets = dets_by_track.get(curr.track_id, [])
        if not curr_dets:
            continue
        curr_first = curr_dets[0]
        c_cx = (curr_first.x1 + curr_first.x2) / 2.0
        c_cy = (curr_first.y1 + curr_first.y2) / 2.0
        c_diag = math.sqrt((curr_first.x2 - curr_first.x1) ** 2 + (curr_first.y2 - curr_first.y1) ** 2)

        for prev in reversed(merged_tracks):
            if prev.class_name != curr.class_name:
                continue

            gap = curr.first_t - prev.last_t
            if 0.0 <= gap <= max_gap_s:
                prev_dets = dets_by_track.get(prev.track_id, [])
                if not prev_dets:
                    continue
                prev_last = prev_dets[-1]
                p_cx = (prev_last.x1 + prev_last.x2) / 2.0
                p_cy = (prev_last.y1 + prev_last.y2) / 2.0

                dist = math.sqrt((c_cx - p_cx) ** 2 + (c_cy - p_cy) ** 2)
                allowed_dist = max(80.0, c_diag * 2.5) + (gap * 150.0)

                if dist <= allowed_dist:
                    id_mapping[curr.track_id] = prev.track_id
                    prev.last_t = max(prev.last_t, curr.last_t)
                    prev.n_frames += curr.n_frames
                    if curr.best_conf > prev.best_conf and curr.color and curr.color != "unknown":
                        prev.color = curr.color
                    prev.best_conf = max(prev.best_conf, curr.best_conf)
                    dets_by_track[prev.track_id].extend(curr_dets)
                    merged = True
                    break
            elif gap > max_gap_s:
                break

        if not merged:
            merged_tracks.append(curr)

    # Remap IDs in raw detections
    for d in detections:
        curr_id = d.track_id
        while curr_id in id_mapping:
            curr_id = id_mapping[curr_id]
        d.track_id = curr_id

    return merged_tracks, detections
