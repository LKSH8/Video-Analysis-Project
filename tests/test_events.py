"""Unit tests for the event engine on hand-made synthetic tracks."""

import json
from cctv.config import EventsConfig
from cctv.db.repository import DetectionRecord, TrackRecord
from cctv.events import generate_events


def test_person_appearing_and_leaving():
    """Verify person_appeared and person_disappeared are produced for a valid track."""
    track = TrackRecord(
        video_id=1,
        track_id=1,
        class_name="person",
        first_t=5.0,
        last_t=25.0,
        n_frames=20,
        best_conf=0.9,
        best_frame_t=10.0,
    )
    detections = [
        DetectionRecord(
            video_id=1, frame=5, t=5.0, track_id=1, class_name="person",
            conf=0.9, x1=100.0, y1=100.0, x2=200.0, y2=400.0,
        ),
        DetectionRecord(
            video_id=1, frame=25, t=25.0, track_id=1, class_name="person",
            conf=0.9, x1=100.0, y1=100.0, x2=200.0, y2=400.0,
        ),
    ]

    events = generate_events(
        tracks=[track],
        detections=detections,
        video_id=1,
        video_duration_s=60.0,
    )

    types = [e.type for e in events]
    assert "person_appeared" in types
    assert "person_disappeared" in types

    app_event = next(e for e in events if e.type == "person_appeared")
    assert app_event.t_start == 5.0
    assert app_event.track_id == 1

    dis_event = next(e for e in events if e.type == "person_disappeared")
    assert dis_event.t_start == 25.0
    assert dis_event.track_id == 1


def test_laptop_vanishes_with_person_overlap():
    """Verify object_disappeared captures nearest person and occlusion when overlapped."""
    laptop_track = TrackRecord(
        video_id=1,
        track_id=10,
        class_name="laptop",
        first_t=0.0,
        last_t=30.0,
        n_frames=30,
        best_conf=0.95,
        best_frame_t=10.0,
    )
    person_track = TrackRecord(
        video_id=1,
        track_id=2,
        class_name="person",
        first_t=15.0,
        last_t=45.0,
        n_frames=30,
        best_conf=0.9,
        best_frame_t=25.0,
    )

    detections = [
        # Laptop at t=0 and t=30
        DetectionRecord(video_id=1, frame=0, t=0.0, track_id=10, class_name="laptop",
                        conf=0.95, x1=200.0, y1=200.0, x2=300.0, y2=300.0),
        DetectionRecord(video_id=1, frame=30, t=30.0, track_id=10, class_name="laptop",
                        conf=0.95, x1=200.0, y1=200.0, x2=300.0, y2=300.0),
        # Person overlapping laptop at t=29.5 and t=30.0
        DetectionRecord(video_id=1, frame=29, t=29.5, track_id=2, class_name="person",
                        conf=0.9, x1=180.0, y1=150.0, x2=320.0, y2=500.0),
        DetectionRecord(video_id=1, frame=30, t=30.0, track_id=2, class_name="person",
                        conf=0.9, x1=180.0, y1=150.0, x2=320.0, y2=500.0),
    ]

    events = generate_events(
        tracks=[laptop_track, person_track],
        detections=detections,
        video_id=1,
        video_duration_s=60.0,
    )

    obj_dis = next(e for e in events if e.type == "object_disappeared")
    assert obj_dis.track_id == 10
    assert obj_dis.class_name == "laptop"
    assert obj_dis.t_start == 30.0

    details = json.loads(obj_dis.details_json)
    assert details["nearest_person_id"] == 2
    assert details["possibly_occluded"] is True


def test_flickering_detection_filtered():
    """Verify an unstable object (< 1.0s) does not create events."""
    # Flickering cup seen only for 0.3s
    flicker_track = TrackRecord(
        video_id=1,
        track_id=99,
        class_name="cup",
        first_t=10.0,
        last_t=10.3,
        n_frames=2,
        best_conf=0.4,
        best_frame_t=10.1,
    )
    detections = [
        DetectionRecord(video_id=1, frame=10, t=10.0, track_id=99, class_name="cup",
                        conf=0.4, x1=50.0, y1=50.0, x2=80.0, y2=90.0),
        DetectionRecord(video_id=1, frame=11, t=10.3, track_id=99, class_name="cup",
                        conf=0.4, x1=50.0, y1=50.0, x2=80.0, y2=90.0),
    ]

    events = generate_events(
        tracks=[flicker_track],
        detections=detections,
        video_id=1,
        video_duration_s=60.0,
    )
    # No object_appeared or object_disappeared for flicker
    assert len(events) == 0


def test_person_loitering_detected():
    """Verify a person staying roughly in place for 12 seconds triggers loitering."""
    # Person with height 300px standing at (x=200, y=100)
    person_track = TrackRecord(
        video_id=1,
        track_id=5,
        class_name="person",
        first_t=0.0,
        last_t=14.0,
        n_frames=15,
        best_conf=0.92,
        best_frame_t=5.0,
    )
    detections = []
    for s in range(15):
        # Small displacement within 10px (radius allowance is 0.5 * 300 = 150px)
        offset = (s % 2) * 5.0
        detections.append(
            DetectionRecord(
                video_id=1,
                frame=s * 10,
                t=float(s),
                track_id=5,
                class_name="person",
                conf=0.92,
                x1=200.0 + offset,
                y1=100.0,
                x2=300.0 + offset,
                y2=400.0,
            )
        )

    events = generate_events(
        tracks=[person_track],
        detections=detections,
        video_id=1,
        video_duration_s=60.0,
    )

    loiter_events = [e for e in events if e.type == "loitering"]
    assert len(loiter_events) >= 1
    le = loiter_events[0]
    assert le.track_id == 5
    assert (le.t_end - le.t_start) >= 10.0


def test_near_interval_merging():
    """Verify that proximity timestamps with small gaps are merged into one sustained near event."""
    person_track = TrackRecord(
        video_id=1, track_id=1, class_name="person",
        first_t=0.0, last_t=20.0, n_frames=20, best_conf=0.9, best_frame_t=5.0
    )
    laptop_track = TrackRecord(
        video_id=1, track_id=10, class_name="laptop",
        first_t=0.0, last_t=20.0, n_frames=20, best_conf=0.9, best_frame_t=5.0
    )

    detections = []
    # Both detected from t=5.0 to 10.0 close to each other
    # Height of person = 400 - 100 = 300. Person box: [100, 100, 200, 400]
    # Laptop box: [220, 200, 300, 280]. Edge distance = 20px (< 300px)
    for t_step in range(5, 11):
        t = float(t_step)
        detections.append(
            DetectionRecord(video_id=1, frame=int(t * 10), t=t, track_id=1,
                            class_name="person", conf=0.9, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
        )
        detections.append(
            DetectionRecord(video_id=1, frame=int(t * 10), t=t, track_id=10,
                            class_name="laptop", conf=0.9, x1=220.0, y1=200.0, x2=300.0, y2=280.0)
        )

    events = generate_events(
        tracks=[person_track, laptop_track],
        detections=detections,
        video_id=1,
        video_duration_s=30.0,
    )

    near_events = [e for e in events if e.type == "near"]
    assert len(near_events) == 1
    ne = near_events[0]
    assert ne.track_id == 1
    assert ne.object_track_id == 10
    assert ne.t_start == 5.0
    assert ne.t_end == 10.0
