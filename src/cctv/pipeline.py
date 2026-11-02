"""Pipeline module: orchestrates Ingest -> Perceive -> Events -> Evidence -> Database."""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from cctv.config import AppConfig, get_config
from cctv.db.repository import Repository, SQLiteRepository
from cctv.events import generate_events
from cctv.evidence import generate_event_evidence, render_full_tracked_video
from cctv.ingest import read_video_metadata
from cctv.perceive import run_perception

logger = logging.getLogger(__name__)


def process_video(
    video_path: str,
    repository: Optional[Repository] = None,
    start_time: Optional[str] = None,
    config: Optional[AppConfig] = None,
    progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
) -> Dict:
    """Orchestrates the complete single-camera CCTV semantic processing pipeline.
    
    Stages:
        1. Ingest: read video properties and register video.
        2. Idempotency: clear previous detections/tracks/events for this video.
        3. Perceive: YOLO11n object detection + ByteTrack + flicker filtering.
        4. Events: rule-based semantic event detection and interval merging.
        5. Evidence: generate annotated keyframes and H.264 web-playable clips.
        6. Store: persist all structured records in database.
    """
    if config is None:
        config = get_config()

    if repository is None:
        repository = SQLiteRepository(config.system.db_path)

    # 1. Ingest
    logger.info("Stage 1/5: Ingesting video metadata from %s", video_path)
    metadata = read_video_metadata(video_path, start_time=start_time)

    # Get or create video row
    video_rec = repository.get_or_create_video(
        path=metadata.path,
        camera_id=config.system.camera_id,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        frame_count=metadata.frame_count,
        duration_s=metadata.duration_s,
        start_wallclock=metadata.start_wallclock,
    )
    video_id = video_rec.id

    # 2. Idempotency cleanup
    logger.info("Ensuring idempotency: clearing previous records for video_id %d", video_id)
    repository.clear_video_data(video_id)
    repository.update_video_status(video_id, "processing")

    # 3. Perceive
    logger.info("Stage 2/5: Running YOLO perception and ByteTrack tracking...")
    def perceive_progress(cur, total, pct):
        if progress_callback:
            progress_callback(cur, total, pct, f"Tracking objects in frame {cur}/{total} ({pct*100:.1f}%)")

    tracks, detections, frame_stats = run_perception(
        video_path=video_path,
        metadata=metadata,
        config=config.perceive,
        frame_stride=config.ingest.frame_stride,
        progress_callback=perceive_progress,
    )

    # Assign real video_id to all records
    for t in tracks:
        t.video_id = video_id
    for d in detections:
        d.video_id = video_id
    for s in frame_stats:
        s.video_id = video_id

    # Store tracks, detections, frame_stats
    logger.info("Storing %d tracks and %d detections...", len(tracks), len(detections))
    repository.insert_tracks(tracks)
    repository.insert_detections(detections)
    repository.insert_frame_stats(frame_stats)

    # 4. Events
    logger.info("Stage 3/5: Extracting semantic events...")
    if progress_callback:
        progress_callback(1, 1, 1.0, "Extracting semantic events from tracks...")
    events = generate_events(
        tracks=tracks,
        detections=detections,
        video_id=video_id,
        camera_id=config.system.camera_id,
        video_duration_s=metadata.duration_s,
        config=config.events,
    )
    logger.info("Extracted %d semantic events", len(events))

    # 5. Evidence
    logger.info("Stage 4/5: Generating evidence (keyframes & H.264 clips)...")
    def evidence_progress(cur, total, desc):
        if progress_callback:
            pct = cur / max(1, total)
            progress_callback(cur, total, pct, desc)

    events = generate_event_evidence(
        video_path=video_path,
        metadata=metadata,
        events=events,
        detections=detections,
        output_base_dir=config.system.evidence_dir,
        config=config.evidence,
        progress_callback=evidence_progress,
    )

    # Store events
    logger.info("Persisting %d events into database...", len(events))
    repository.insert_events(events)

    # 6. Render full tracked video
    logger.info("Stage 5/5: Rendering full processed video with trackings...")
    tracked_output_path = Path(config.system.evidence_dir) / str(video_id) / "full_tracked.mp4"
    def full_track_progress(cur, total, desc):
        if progress_callback:
            pct = cur / max(1, total)
            progress_callback(cur, total, pct, desc)

    render_full_tracked_video(
        video_path=video_path,
        metadata=metadata,
        detections=detections,
        output_path=tracked_output_path,
        frame_stride=config.ingest.frame_stride,
        progress_callback=full_track_progress,
    )

    # Mark video completed
    processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    repository.update_video_status(video_id, "completed", processed_at=processed_at)

    logger.info("Processing complete for video %d!", video_id)
    return {
        "video_id": video_id,
        "path": metadata.path,
        "duration_s": metadata.duration_s,
        "n_tracks": len(tracks),
        "n_detections": len(detections),
        "n_events": len(events),
        "tracked_video_path": str(tracked_output_path),
        "events_by_type": repository.get_event_counts_by_type(video_id),
    }
