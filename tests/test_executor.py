"""Unit tests for QueryExecutor and AnswerBuilder using seeded in-memory DB."""

from cctv.query.answer import build_answer
from cctv.query.executor import QueryExecutor
from cctv.query.schema import AnchorEvent, QuerySpec, TemporalRelation, TimeRange


def test_executor_anchor_before_logic(seeded_repo):
    """Test 'who was near laptop before it disappeared' resolves laptop disappearance anchor."""
    executor = QueryExecutor(seeded_repo)

    spec = QuerySpec(
        intent="who_near",
        event_types=["near"],
        classes=["person", "laptop"],
        temporal_relation=TemporalRelation(
            relation="before",
            anchor=AnchorEvent(event_type="object_disappeared", class_name="laptop"),
            window_s=30.0,
        ),
        raw_question="Who was near the laptop before it disappeared?",
    )

    result = executor.execute(video_id=1, spec=spec)
    assert result.anchor_event is not None
    assert result.anchor_event.t_start == 35.0  # Laptop disappeared at 35.0s
    assert len(result.events) == 1  # Person 2 near event was 20.0s - 32.0s
    assert result.events[0].track_id == 2

    # Check answer summary
    answer = build_answer(result)
    assert answer.has_results is True
    assert "Person 2" in answer.summary
    assert "before it object disappeared" in answer.summary
    assert len(answer.evidence) == 1


def test_executor_count_people(seeded_repo):
    """Test counting people returns distinct tracks from database."""
    executor = QueryExecutor(seeded_repo)
    spec = QuerySpec(
        intent="count",
        event_types=[],
        classes=["person"],
        raw_question="How many people are there?",
    )
    result = executor.execute(video_id=1, spec=spec)
    assert result.total_count == 2  # Person 1 and Person 2

    answer = build_answer(result)
    assert "Found a total of 2 person(s)" in answer.summary


def test_executor_time_range_filter(seeded_repo):
    """Test filtering events strictly within [10.0, 30.0]."""
    executor = QueryExecutor(seeded_repo)
    spec = QuerySpec(
        intent="what_happened",
        event_types=[],
        classes=[],
        time_range=TimeRange(start_s=10.0, end_s=30.0),
        raw_question="What happened between 10 and 30 seconds?",
    )
    result = executor.execute(video_id=1, spec=spec)
    assert len(result.events) > 0
    # Every event returned should intersect or be within 10.0 and 30.0
    for ev in result.events:
        assert ev.t_end >= 10.0
        assert ev.t_start <= 30.0


def test_executor_empty_result_case(seeded_repo):
    """Test querying non-existent class/track returns clean empty response without crashing."""
    executor = QueryExecutor(seeded_repo)
    spec = QuerySpec(
        intent="list_events",
        event_types=["person_appeared"],
        track_id=999,  # does not exist
        raw_question="Show person 999",
    )
    result = executor.execute(video_id=1, spec=spec)
    assert len(result.events) == 0

    answer = build_answer(result)
    assert answer.has_results is False
    assert "No matching events were found" in answer.summary


def test_executor_anchor_not_found(seeded_repo):
    """Test temporal relation where anchor event does not exist."""
    executor = QueryExecutor(seeded_repo)
    spec = QuerySpec(
        intent="who_near",
        event_types=["near"],
        temporal_relation=TemporalRelation(
            relation="before",
            anchor=AnchorEvent(event_type="object_disappeared", class_name="cell phone"),  # No phone disappeared
            window_s=30.0,
        ),
        raw_question="Who was near the phone before it disappeared?",
    )
    result = executor.execute(video_id=1, spec=spec)
    assert result.anchor_event is None
    assert len(result.events) == 0

    answer = build_answer(result)
    assert answer.has_results is False
    assert "did not occur" in answer.summary


def test_executor_color_query(seeded_repo):
    """Test filtering tracks by color attribute."""
    from cctv.db.repository import TrackRecord
    seeded_repo.insert_tracks([
        TrackRecord(video_id=1, track_id=5, class_name="car", first_t=5.0, last_t=15.0, n_frames=30, best_conf=0.92, best_frame_t=10.0, color="red"),
        TrackRecord(video_id=1, track_id=6, class_name="car", first_t=20.0, last_t=30.0, n_frames=30, best_conf=0.88, best_frame_t=25.0, color="blue"),
    ])

    executor = QueryExecutor(seeded_repo)
    spec_red = QuerySpec(
        intent="count",
        classes=["car"],
        color="red",
        raw_question="How many red cars were detected?",
    )
    res_red = executor.execute(video_id=1, spec=spec_red)
    assert res_red.total_count == 1
    ans_red = build_answer(res_red)
    assert "1 red car(s)" in ans_red.summary

    spec_none = QuerySpec(
        intent="list_events",
        classes=["car"],
        color="green",
        raw_question="Did you see any green car?",
    )
    res_none = executor.execute(video_id=1, spec=spec_none)
    assert len(res_none.events) == 0
    ans_none = build_answer(res_none)
    assert "No green car was detected" in ans_none.summary
