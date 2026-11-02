"""Event engine: converts tracks and detections into semantic CCTV events."""

from collections import defaultdict
import json
import logging
import math
from typing import Dict, List, Optional, Tuple

from cctv.config import EventsConfig, get_config
from cctv.db.repository import DetectionRecord, EventRecord, TrackRecord

logger = logging.getLogger(__name__)


def box_edge_distance(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Calculate Euclidean edge-to-edge distance between two 2D boxes (x1, y1, x2, y2).
    
    Returns 0.0 if boxes overlap.
    """
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    if x2_a < x1_b:
        dx = x1_b - x2_a
    elif x2_b < x1_a:
        dx = x1_a - x2_b
    else:
        dx = 0.0

    if y2_a < y1_b:
        dy = y1_b - y2_a
    elif y2_b < y1_a:
        dy = y1_a - y2_b
    else:
        dy = 0.0

    return math.sqrt(dx * dx + dy * dy)


def box_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def box_coverage(
    subject_box: Tuple[float, float, float, float],
    covering_box: Tuple[float, float, float, float],
) -> float:
    """Calculate fraction of subject_box covered by covering_box."""
    x1_a, y1_a, x2_a, y2_a = subject_box
    x1_b, y1_b, x2_b, y2_b = covering_box

    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    subject_area = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    if subject_area <= 0:
        return 0.0
    return inter_area / subject_area


def generate_events(
    tracks: List[TrackRecord],
    detections: List[DetectionRecord],
    video_id: int = 0,
    camera_id: str = "cam1",
    video_duration_s: Optional[float] = None,
    config: Optional[EventsConfig] = None,
) -> List[EventRecord]:
    """Extract semantic events from tracked objects and detections.
    
    Generates:
        - person_appeared
        - person_disappeared
        - object_appeared
        - object_disappeared
        - near
        - loitering
    """
    if config is None:
        config = get_config().events

    events: List[EventRecord] = []

    # Index detections by track_id and by frame/time
    dets_by_track: Dict[int, List[DetectionRecord]] = defaultdict(list)
    dets_by_time: Dict[float, List[DetectionRecord]] = defaultdict(list)
    tracks_by_id: Dict[int, TrackRecord] = {t.track_id: t for t in tracks}

    for d in detections:
        dets_by_track[d.track_id].append(d)
        dets_by_time[round(d.t, 2)].append(d)

    # Sort detections within each track by timestamp
    for tid in dets_by_track:
        dets_by_track[tid].sort(key=lambda d: d.t)

    # 1. Person Events (appeared, disappeared, loitering)
    for track in tracks:
        if track.class_name != "person":
            continue

        track_dets = dets_by_track.get(track.track_id, [])
        first_box = [track_dets[0].x1, track_dets[0].y1, track_dets[0].x2, track_dets[0].y2] if track_dets else []
        last_box = [track_dets[-1].x1, track_dets[-1].y1, track_dets[-1].x2, track_dets[-1].y2] if track_dets else []

        # person_appeared
        events.append(
            EventRecord(
                video_id=video_id,
                camera_id=camera_id,
                type="person_appeared",
                t_start=track.first_t,
                t_end=track.first_t,
                track_id=track.track_id,
                class_name="person",
                confidence=track.best_conf,
                details_json=json.dumps({"box": first_box, "frame": track_dets[0].frame if track_dets else 0}),
            )
        )

        # person_disappeared: person leaves the scene
        # Only trigger if track ended before the video finished (e.g. > 0.5s before video ends)
        is_at_end = (
            video_duration_s is not None and (video_duration_s - track.last_t) <= 0.5
        )
        if not is_at_end:
            events.append(
                EventRecord(
                    video_id=video_id,
                    camera_id=camera_id,
                    type="person_disappeared",
                    t_start=track.last_t,
                    t_end=track.last_t,
                    track_id=track.track_id,
                    class_name="person",
                    confidence=track.best_conf,
                    details_json=json.dumps({"box": last_box, "frame": track_dets[-1].frame if track_dets else 0}),
                )
            )

        # loitering: centroid displacement < loiter_radius_ratio * person_height for >= loiter_seconds
        if len(track_dets) >= 2 and (track.last_t - track.first_t) >= config.loiter_seconds:
            loiter_events = _detect_loitering(track, track_dets, video_id, camera_id, config)
            events.extend(loiter_events)

    # 2. Object Events (object_appeared, object_disappeared)
    for track in tracks:
        if track.class_name == "person":
            continue

        duration = track.last_t - track.first_t
        if duration < config.object_stable_seconds:
            # Drop non-stable object
            continue

        track_dets = dets_by_track.get(track.track_id, [])
        first_box = [track_dets[0].x1, track_dets[0].y1, track_dets[0].x2, track_dets[0].y2] if track_dets else []
        last_box = [track_dets[-1].x1, track_dets[-1].y1, track_dets[-1].x2, track_dets[-1].y2] if track_dets else []

        # object_appeared
        events.append(
            EventRecord(
                video_id=video_id,
                camera_id=camera_id,
                type="object_appeared",
                t_start=track.first_t,
                t_end=track.first_t,
                track_id=track.track_id,
                class_name=track.class_name,
                confidence=track.best_conf,
                details_json=json.dumps({"box": first_box, "frame": track_dets[0].frame if track_dets else 0}),
            )
        )

        # object_disappeared:
        # not detected for >= object_missing_seconds (default 3s)
        has_vanished = False
        if video_duration_s is not None:
            if (video_duration_s - track.last_t) >= config.object_missing_seconds:
                has_vanished = True
        else:
            # In synthetic mode without video_duration_s, if track ended
            has_vanished = True

        if has_vanished:
            disappear_details = _analyze_object_disappearance(
                track=track,
                last_box=last_box,
                detections=detections,
                tracks_by_id=tracks_by_id,
                config=config,
            )
            events.append(
                EventRecord(
                    video_id=video_id,
                    camera_id=camera_id,
                    type="object_disappeared",
                    t_start=track.last_t,
                    t_end=track.last_t,
                    track_id=track.track_id,
                    class_name=track.class_name,
                    confidence=track.best_conf,
                    details_json=json.dumps(disappear_details),
                )
            )

    # 3. Near Events (Person-Object proximity intervals)
    near_events = _detect_near_events(
        tracks=tracks,
        dets_by_track=dets_by_track,
        video_id=video_id,
        camera_id=camera_id,
        config=config,
    )
    events.extend(near_events)

    # Sort all events chronologically by t_start
    events.sort(key=lambda e: (e.t_start, e.t_end, e.type))
    return events


def _detect_loitering(
    track: TrackRecord,
    track_dets: List[DetectionRecord],
    video_id: int,
    camera_id: str,
    config: EventsConfig,
) -> List[EventRecord]:
    """Detect intervals where a person stays within a radius for >= loiter_seconds."""
    events: List[EventRecord] = []
    n = len(track_dets)
    if n < 2:
        return events

    # Precalculate centroids and heights
    centroids: List[Tuple[float, float, float, float]] = []  # (t, cx, cy, h)
    for d in track_dets:
        cx = (d.x1 + d.x2) / 2.0
        cy = (d.y1 + d.y2) / 2.0
        h = max(1.0, d.y2 - d.y1)
        centroids.append((d.t, cx, cy, h))

    # Sliding window anchor
    i = 0
    while i < n:
        anchor_t, anchor_cx, anchor_cy, anchor_h = centroids[i]
        radius = config.loiter_radius_ratio * anchor_h
        
        j = i + 1
        max_disp = 0.0
        while j < n:
            curr_t, curr_cx, curr_cy, _ = centroids[j]
            disp = math.sqrt((curr_cx - anchor_cx) ** 2 + (curr_cy - anchor_cy) ** 2)
            if disp > radius:
                break
            max_disp = max(max_disp, disp)
            j += 1

        duration = centroids[j - 1][0] - anchor_t
        if duration >= config.loiter_seconds:
            events.append(
                EventRecord(
                    video_id=video_id,
                    camera_id=camera_id,
                    type="loitering",
                    t_start=round(anchor_t, 2),
                    t_end=round(centroids[j - 1][0], 2),
                    track_id=track.track_id,
                    class_name="person",
                    confidence=track.best_conf,
                    details_json=json.dumps({
                        "duration_s": round(duration, 2),
                        "max_displacement_px": round(max_disp, 1),
                        "allowed_radius_px": round(radius, 1),
                    }),
                )
            )
            i = j
        else:
            i += 1

    return events


def _analyze_object_disappearance(
    track: TrackRecord,
    last_box: List[float],
    detections: List[DetectionRecord],
    tracks_by_id: Dict[int, TrackRecord],
    config: EventsConfig,
) -> Dict:
    """Analyze the window around object disappearance for nearest person and occlusion."""
    if not last_box:
        return {"last_box": [], "nearest_person_id": None, "possibly_occluded": False}

    obj_box_tuple = (last_box[0], last_box[1], last_box[2], last_box[3])
    lookback_t = track.last_t - config.object_lookback_seconds

    # Look for person detections near object's last known time
    candidate_persons: Dict[int, List[DetectionRecord]] = defaultdict(list)
    for d in detections:
        if d.class_name == "person" and (lookback_t <= d.t <= track.last_t + 0.5):
            candidate_persons[d.track_id].append(d)

    nearest_person_id: Optional[int] = None
    min_dist = float("inf")
    possibly_occluded = False

    for pid, p_dets in candidate_persons.items():
        for pd in p_dets:
            p_box = (pd.x1, pd.y1, pd.x2, pd.y2)
            p_h = max(1.0, pd.y2 - pd.y1)
            dist = box_edge_distance(p_box, obj_box_tuple)
            cov = box_coverage(obj_box_tuple, p_box)

            if cov >= config.occlusion_iou_threshold:
                possibly_occluded = True

            if dist < min_dist:
                min_dist = dist
                if dist <= config.near_px_ratio * p_h:
                    nearest_person_id = pid

    return {
        "last_box": last_box,
        "nearest_person_id": nearest_person_id,
        "possibly_occluded": possibly_occluded,
        "min_distance_px": round(min_dist, 1) if min_dist != float("inf") else None,
    }


def _detect_near_events(
    tracks: List[TrackRecord],
    dets_by_track: Dict[int, List[DetectionRecord]],
    video_id: int,
    camera_id: str,
    config: EventsConfig,
) -> List[EventRecord]:
    """Detect sustained proximity between people and objects."""
    events: List[EventRecord] = []
    person_tracks = [t for t in tracks if t.class_name == "person"]
    object_tracks = [t for t in tracks if t.class_name != "person"]

    for p in person_tracks:
        p_dets = dets_by_track.get(p.track_id, [])
        if not p_dets:
            continue
        # Map time to person detection (round t to 2 decimal places)
        p_by_t = {round(d.t, 2): d for d in p_dets}

        for obj in object_tracks:
            obj_dets = dets_by_track.get(obj.track_id, [])
            if not obj_dets:
                continue
            obj_by_t = {round(d.t, 2): d for d in obj_dets}

            # Find common timestamps
            common_times = sorted(set(p_by_t.keys()).intersection(obj_by_t.keys()))
            if not common_times:
                continue

            # Identify close timestamps
            close_times: List[float] = []
            min_dist_observed = float("inf")

            for t in common_times:
                pd = p_by_t[t]
                od = obj_by_t[t]
                p_box = (pd.x1, pd.y1, pd.x2, pd.y2)
                o_box = (od.x1, od.y1, od.x2, od.y2)
                p_h = max(1.0, pd.y2 - pd.y1)

                dist = box_edge_distance(p_box, o_box)
                threshold = config.near_ratio * p_h

                if dist < threshold:
                    close_times.append(t)
                    min_dist_observed = min(min_dist_observed, dist)

            if not close_times:
                continue

            # Merge contiguous timestamps into intervals (allow up to 1.0s gap for tracking jitter)
            intervals: List[Tuple[float, float]] = []
            curr_start = close_times[0]
            curr_end = close_times[0]

            for t in close_times[1:]:
                if (t - curr_end) <= 1.05:  # allow 1.0s jitter
                    curr_end = t
                else:
                    if (curr_end - curr_start) >= config.near_min_seconds:
                        intervals.append((curr_start, curr_end))
                    curr_start = t
                    curr_end = t

            if (curr_end - curr_start) >= config.near_min_seconds:
                intervals.append((curr_start, curr_end))

            for t_start, t_end in intervals:
                events.append(
                    EventRecord(
                        video_id=video_id,
                        camera_id=camera_id,
                        type="near",
                        t_start=round(t_start, 2),
                        t_end=round(t_end, 2),
                        track_id=p.track_id,
                        object_track_id=obj.track_id,
                        class_name="person",
                        object_class=obj.class_name,
                        confidence=round(min(p.best_conf, obj.best_conf), 2),
                        details_json=json.dumps({
                            "duration_s": round(t_end - t_start, 2),
                            "min_distance_px": round(min_dist_observed, 1),
                        }),
                    )
                )

    return events
