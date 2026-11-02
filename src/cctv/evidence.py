"""Evidence generation module: extracts annotated keyframes and H.264 playable clips."""

from collections import defaultdict
import logging
from pathlib import Path
import subprocess
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import imageio_ffmpeg
import numpy as np

from cctv.config import EvidenceConfig, get_config
from cctv.db.repository import DetectionRecord, EventRecord
from cctv.ingest import VideoMetadata

logger = logging.getLogger(__name__)


def annotate_frame(
    frame: np.ndarray,
    detections: List[DetectionRecord],
    highlight_track_ids: List[int],
    timestamp_text: str,
    event_title: Optional[str] = None,
    camera_id: str = "CAM-01",
    frame_idx: Optional[int] = None,
) -> np.ndarray:
    """Draw forensic bounding boxes, label tabs, corner ticks, and overlay telemetry."""
    img = frame.copy()
    h, w = img.shape[:2]

    # DESIGN.md Token Palette in BGR
    # Amber #D9A441, Steel #6E8CA3, Green #5E9C76, Red #C8453B, Inset #0A0B0C
    c_person = (65, 164, 217)   # Amber
    c_vehicle = (163, 140, 110) # Steel
    c_object = (118, 156, 94)   # Green
    c_flagged = (59, 69, 200)   # Red
    c_inverse = (16, 15, 14)    # Text inverse #0E0F10
    c_tick = (100, 96, 92)      # Hairline border

    def get_color(class_name: str, is_highlight: bool):
        if is_highlight:
            return c_flagged
        if class_name == "person":
            return c_person
        if class_name in ("car", "truck", "bus", "motorcycle"):
            return c_vehicle
        return c_object

    # 1. 10px Corner Registration Ticks
    tick_len = 12
    # Top-Left
    cv2.line(img, (6, 6), (6 + tick_len, 6), c_tick, 1)
    cv2.line(img, (6, 6), (6, 6 + tick_len), c_tick, 1)
    # Top-Right
    cv2.line(img, (w - 7, 6), (w - 7 - tick_len, 6), c_tick, 1)
    cv2.line(img, (w - 7, 6), (w - 7, 6 + tick_len), c_tick, 1)
    # Bottom-Left
    cv2.line(img, (6, h - 7), (6 + tick_len, h - 7), c_tick, 1)
    cv2.line(img, (6, h - 7), (6, h - 7 - tick_len), c_tick, 1)
    # Bottom-Right
    cv2.line(img, (w - 7, h - 7), (w - 7 - tick_len, h - 7), c_tick, 1)
    cv2.line(img, (w - 7, h - 7), (w - 7, h - 7 - tick_len), c_tick, 1)

    # 2. Detection Bounding Boxes and Label Tabs
    for d in detections:
        is_highlight = d.track_id in highlight_track_ids
        color = get_color(d.class_name, is_highlight)

        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        # 1px solid stroke, sharp corners
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # Tab text: CLASS CONF · TRK-XXXX
        flag_suffix = " · FLAG" if is_highlight else ""
        label = f"{d.class_name.upper()} {d.conf:.2f} · TRK-{d.track_id:04d}{flag_suffix}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tab_h = th + 6
        tab_w = tw + 8
        tab_y1 = max(0, y1 - tab_h)
        tab_y2 = tab_y1 + tab_h

        # Solid class color banner
        cv2.rectangle(img, (x1, tab_y1), (x1 + tab_w, tab_y2), color, -1)
        # Text in dark inverse #0E0F10
        cv2.putText(
            img, label, (x1 + 4, tab_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, c_inverse, 1, cv2.LINE_AA
        )

    # 3. Burned-in Telemetry Overlays
    # Top-Left Telemetry
    f_str = f" · F{frame_idx:06d}" if frame_idx is not None else ""
    event_str = f" · [{event_title.upper()}]" if event_title else ""
    tl_text = f"{camera_id} · {timestamp_text}{f_str}{event_str}"

    (tw_tl, th_tl), _ = cv2.getTextSize(tl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(img, (10, 10), (18 + tw_tl, 28), (14, 15, 16), -1)
    cv2.rectangle(img, (10, 10), (18 + tw_tl, 28), c_tick, 1)
    cv2.putText(
        img, tl_text, (14, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (65, 164, 217), 1, cv2.LINE_AA  # Amber text
    )

    # Bottom-Right Telemetry: Dimensions
    br_text = f"{w}x{h}"
    (tw_br, th_br), _ = cv2.getTextSize(br_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(img, (w - tw_br - 18, h - 26), (w - 10, h - 10), (14, 15, 16), -1)
    cv2.rectangle(img, (w - tw_br - 18, h - 26), (w - 10, h - 10), c_tick, 1)
    cv2.putText(
        img, br_text, (w - tw_br - 14, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 144, 140), 1, cv2.LINE_AA
    )

    return img


def format_seconds_to_time(seconds: float) -> str:
    """Format seconds into MM:SS.ss."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def generate_event_evidence(
    video_path: str,
    metadata: VideoMetadata,
    events: List[EventRecord],
    detections: List[DetectionRecord],
    output_base_dir: str = "data/evidence",
    config: Optional[EvidenceConfig] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[EventRecord]:
    """Generate annotated keyframe JPGs and H.264 MP4 clips for each event."""
    if config is None:
        config = get_config().evidence

    if not events:
        return []

    # Map detections by frame and by time
    dets_by_t: Dict[float, List[DetectionRecord]] = defaultdict(list)
    for d in detections:
        dets_by_t[round(d.t, 2)].append(d)

    # Ensure evidence subdirectories exist
    video_id = events[0].video_id
    video_evidence_dir = Path(output_base_dir) / str(video_id)
    keyframes_dir = video_evidence_dir / "keyframes"
    clips_dir = video_evidence_dir / "clips"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file for evidence extraction: %s", video_path)
        return events

    fps = metadata.fps if metadata.fps > 0 else 30.0

    updated_events: List[EventRecord] = []

    for idx, ev in enumerate(events):
        event_num = idx + 1
        if progress_callback:
            progress_callback(event_num, len(events), f"Encoding clip {event_num}/{len(events)}: {ev.type} (#{ev.track_id})")
        highlight_ids = [ev.track_id]
        if ev.object_track_id is not None:
            highlight_ids.append(ev.object_track_id)

        # 1. Keyframe extraction
        key_t = ev.t_start if ev.t_start == ev.t_end else (ev.t_start + ev.t_end) / 2.0
        frame_no = int(key_t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()

        keyframe_rel_path = f"{output_base_dir}/{video_id}/keyframes/event_{event_num}_{ev.type}.jpg"
        keyframe_full_path = keyframes_dir / f"event_{event_num}_{ev.type}.jpg"

        if ret and frame is not None:
            # Find closest detections
            close_dets: List[DetectionRecord] = []
            for t_val, dt_list in dets_by_t.items():
                if abs(t_val - key_t) <= 0.5:
                    close_dets.extend(dt_list)
            # Deduplicate by track_id
            seen_tracks = set()
            dedup_dets = []
            for d in close_dets:
                if d.track_id not in seen_tracks:
                    seen_tracks.add(d.track_id)
                    dedup_dets.append(d)

            annotated_kf = annotate_frame(
                frame=frame,
                detections=dedup_dets,
                highlight_track_ids=highlight_ids,
                timestamp_text=format_seconds_to_time(key_t),
                event_title=ev.type,
            )
            cv2.imwrite(str(keyframe_full_path), annotated_kf)
            ev.keyframe_path = keyframe_rel_path
        else:
            logger.warning("Could not read keyframe at %.2fs for event %d", key_t, event_num)

        # 2. Clip extraction
        t_clip_start = max(0.0, ev.t_start - config.pre_roll_seconds)
        t_clip_end = min(metadata.duration_s, ev.t_end + config.post_roll_seconds)
        if (t_clip_end - t_clip_start) > config.max_clip_seconds:
            t_clip_end = t_clip_start + config.max_clip_seconds

        clip_rel_path = f"{output_base_dir}/{video_id}/clips/event_{event_num}_{ev.type}.mp4"
        clip_full_path = clips_dir / f"event_{event_num}_{ev.type}.mp4"

        _extract_and_encode_clip(
            cap=cap,
            ffmpeg_exe=ffmpeg_exe,
            fps=fps,
            t_start=t_clip_start,
            t_end=t_clip_end,
            output_path=clip_full_path,
            dets_by_t=dets_by_t,
            highlight_ids=highlight_ids,
            event_type=ev.type,
            width=metadata.width,
            height=metadata.height,
        )
        ev.clip_path = clip_rel_path

        updated_events.append(ev)

    cap.release()
    return updated_events


def _extract_and_encode_clip(
    cap: cv2.VideoCapture,
    ffmpeg_exe: str,
    fps: float,
    t_start: float,
    t_end: float,
    output_path: Path,
    dets_by_t: Dict[float, List[DetectionRecord]],
    highlight_ids: List[int],
    event_type: str,
    width: int,
    height: int,
) -> None:
    """Extracts frames, annotates them, and pipes them directly to ffmpeg for H.264 encoding."""
    start_frame = int(t_start * fps)
    end_frame = int(t_end * fps)

    if end_frame <= start_frame:
        end_frame = start_frame + int(fps * 2.0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Setup ffmpeg process
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-crf", "23",
        str(output_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        curr_frame = start_frame
        while curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_t = float(curr_frame) / fps
            # Grab close detections
            frame_dets = []
            for dt_val, d_list in dets_by_t.items():
                if abs(dt_val - current_t) <= (0.5 / fps):
                    frame_dets = d_list
                    break

            annotated = annotate_frame(
                frame=frame,
                detections=frame_dets,
                highlight_track_ids=highlight_ids,
                timestamp_text=format_seconds_to_time(current_t),
                event_title=event_type,
            )

            # Write raw bytes to ffmpeg stdin
            if proc.stdin:
                proc.stdin.write(annotated.tobytes())

            curr_frame += 1

        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=30)
    except Exception as e:
        logger.error("FFmpeg clip generation error: %s", e)


def render_full_tracked_video(
    video_path: str,
    metadata: VideoMetadata,
    detections: List[DetectionRecord],
    output_path: Path | str,
    frame_stride: int = 1,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """Renders the entire video with all bounding boxes and track IDs burned in as H.264 MP4."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Group detections by frame number
    dets_by_frame: Dict[int, List[DetectionRecord]] = defaultdict(list)
    for d in detections:
        dets_by_frame[d.frame].append(d)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    fps = metadata.fps if metadata.fps > 0 else 30.0
    stride = max(1, frame_stride)
    # Effective output FPS so video plays at normal 1x speed
    out_fps = max(1.0, fps / stride)
    total_frames = metadata.frame_count

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{metadata.width}x{metadata.height}",
        "-pix_fmt", "bgr24",
        "-r", str(out_fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-crf", "23",
        str(out_file),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        frame_idx = 0
        written_count = 0
        expected_writes = total_frames // stride

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % stride == 0:
                current_t = float(frame_idx) / fps
                frame_dets = dets_by_frame.get(frame_idx, [])

                annotated = annotate_frame(
                    frame=frame,
                    detections=frame_dets,
                    highlight_track_ids=[],
                    timestamp_text=format_seconds_to_time(current_t),
                    event_title="",
                    camera_id="CAM-01",
                    frame_idx=frame_idx,
                )

                if proc.stdin:
                    proc.stdin.write(annotated.tobytes())
                written_count += 1

                if progress_callback and (written_count % 30 == 0 or frame_idx >= total_frames - stride):
                    progress_callback(
                        written_count,
                        expected_writes,
                        f"Rendering full tracked video ({written_count}/{expected_writes} frames)...",
                    )

            frame_idx += 1

        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=180)
    finally:
        cap.release()

    return str(out_file)
