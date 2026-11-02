"""Pytest fixtures for Semantic CCTV test suite."""

import json
from pathlib import Path
import pytest

from cctv.db.repository import (
    DetectionRecord,
    EventRecord,
    FrameStatRecord,
    SQLiteRepository,
    TrackRecord,
)


@pytest.fixture
def memory_repo() -> SQLiteRepository:
    """Provides a fresh, empty in-memory SQLite repository."""
    schema_path = str(Path(__file__).parent.parent / "src" / "cctv" / "db" / "schema.sql")
    repo = SQLiteRepository(db_path=":memory:", schema_path=schema_path)
    return repo


@pytest.fixture
def seeded_repo(memory_repo: SQLiteRepository) -> SQLiteRepository:
    """Provides a seeded in-memory SQLite repository with synthetic incident data.
    
    Incident Scenario:
    - Video 1: 60 seconds duration
    - Laptop (track 10): present from 0.0s to 35.0s, then disappears at 35.0s
    - Person 1: enters at 5.0s, exits at 25.0s
    - Person 2: enters at 10.0s, loiters 15.0-28.0s, near laptop 20.0-32.0s, exits at 40.0s
    - Backpack (track 20): present 15.0-50.0s
    """
    repo = memory_repo

    # Insert Video
    video = repo.get_or_create_video(
        path="data/videos/synthetic_test.mp4",
        camera_id="cam1",
        fps=30.0,
        width=1280,
        height=720,
        frame_count=1800,
        duration_s=60.0,
        start_wallclock="2026-11-01 10:00:00",
    )
    video_id = video.id

    # Insert Tracks
    tracks = [
        TrackRecord(
            video_id=video_id,
            track_id=1,
            class_name="person",
            first_t=5.0,
            last_t=25.0,
            n_frames=600,
            best_conf=0.88,
            best_frame_t=12.0,
        ),
        TrackRecord(
            video_id=video_id,
            track_id=2,
            class_name="person",
            first_t=10.0,
            last_t=40.0,
            n_frames=900,
            best_conf=0.92,
            best_frame_t=25.0,
        ),
        TrackRecord(
            video_id=video_id,
            track_id=10,
            class_name="laptop",
            first_t=0.0,
            last_t=35.0,
            n_frames=1050,
            best_conf=0.95,
            best_frame_t=10.0,
        ),
        TrackRecord(
            video_id=video_id,
            track_id=20,
            class_name="backpack",
            first_t=15.0,
            last_t=50.0,
            n_frames=1050,
            best_conf=0.85,
            best_frame_t=20.0,
        ),
    ]
    repo.insert_tracks(tracks)

    # Insert Events
    events = [
        EventRecord(
            video_id=video_id,
            camera_id="cam1",
            type="object_appeared",
            t_start=0.0,
            t_end=0.0,
            track_id=10,
            class_name="laptop",
            confidence=0.95,
            details_json=json.dumps({"box": [200, 300, 350, 420]}),
            keyframe_path="data/evidence/1/keyframes/event_1.jpg",
            clip_path="data/evidence/1/clips/event_1.mp4",
        ),
        EventRecord(
            video_id=video_id,
            camera_id="cam1",
            type="person_appeared",
            t_start=5.0,
            t_end=5.0,
            track_id=1,
            class_name="person",
            confidence=0.88,
            details_json=json.dumps({"box": [50, 100, 150, 380]}),
            keyframe_path="data/evidence/1/keyframes/event_2.jpg",
            clip_path="data/evidence/1/clips/event_2.mp4",
        ),
        EventRecord(
            video_id=video_id,
            camera_id="cam1",
            type="person_appeared",
            t_start=10.0,
            t_end=10.0,
            track_id=2,
            class_name="person",
            confidence=0.92,
            details_json=json.dumps({"box": [500, 120, 600, 410]}),
            keyframe_path="data/evidence/1/keyframes/event_3.jpg",
            clip_path="data/evidence/1/clips/event_3.mp4",
        ),
        EventRecord(
            video_id=video_id,
            camera_id="cam1",
            type="loitering",
            t_start=15.0,
            t_end=28.0,
            track_id=2,
            class_name="person",
            confidence=0.90,
            details_json=json.dumps({"duration": 13.0, "displacement": 12.5}),
            keyframe_path="data/evidence/1/keyframes/event_4.jpg",
            clip_path="data/evidence/1/clips/event_4.mp4",
        ),
        EventRecord(
            video_id=video_id,
            camera_id="cam1",
            type="near",
            t_start=20.0,
            t_end=32.0,
            track_id=2,
            object_track_id=10,
            class_name="person",
            object_class="laptop",
            confidence=0.89,
            details_json=json.dumps({"min_distance_px": 15.0, "duration": 12.0}),
            keyframe_path="data/evidence/1/keyframes/event_5.jpg",
            clip_path="data/evidence/1/clips/event_5.mp4",
        ),
        EventRecord(
            video_id=video_id,
            camera_id="cam1",
            type="person_disappeared",
            t_start=25.0,
            t_end=25.0,
            track_id=1,
            class_name="person",
            confidence=0.88,
            details_json=json.dumps({}),
            keyframe_path="data/evidence/1/keyframes/event_6.jpg",
            clip_path="data/evidence/1/clips/event_6.mp4",
        ),
        EventRecord(
            video_id=video_id,
            camera_id="cam1",
            type="object_disappeared",
            t_start=35.0,
            t_end=35.0,
            track_id=10,
            class_name="laptop",
            confidence=0.91,
            details_json=json.dumps({
                "nearest_person_id": 2,
                "possibly_occluded": False,
                "last_box": [200, 300, 350, 420],
            }),
            keyframe_path="data/evidence/1/keyframes/event_7.jpg",
            clip_path="data/evidence/1/clips/event_7.mp4",
        ),
        EventRecord(
            video_id=video_id,
            camera_id="cam1",
            type="person_disappeared",
            t_start=40.0,
            t_end=40.0,
            track_id=2,
            class_name="person",
            confidence=0.92,
            details_json=json.dumps({}),
            keyframe_path="data/evidence/1/keyframes/event_8.jpg",
            clip_path="data/evidence/1/clips/event_8.mp4",
        ),
    ]
    repo.insert_events(events)

    # Insert Frame Stats
    frame_stats = []
    for sec in range(61):
        t = float(sec)
        # People present
        p_count = 0
        if 5.0 <= t <= 25.0:
            p_count += 1
        if 10.0 <= t <= 40.0:
            p_count += 1

        # Objects present
        o_count = 0
        if 0.0 <= t <= 35.0:
            o_count += 1  # laptop
        if 15.0 <= t <= 50.0:
            o_count += 1  # backpack

        frame_stats.append(
            FrameStatRecord(
                video_id=video_id,
                t=t,
                n_people=p_count,
                n_objects=o_count,
            )
        )
    repo.insert_frame_stats(frame_stats)

    return repo
