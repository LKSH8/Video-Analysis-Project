"""Unit tests for the SQLite repository layer."""

def test_seeded_repo_has_data(seeded_repo):
    video = seeded_repo.get_video(1)
    assert video is not None
    assert video.fps == 30.0
    assert video.duration_s == 60.0

    tracks = seeded_repo.get_tracks(1)
    assert len(tracks) == 4

    events = seeded_repo.query_events(1)
    assert len(events) == 8

    # Test count queries
    assert seeded_repo.count_tracks(1, "person") == 2
    assert seeded_repo.count_tracks(1, "laptop") == 1
    assert seeded_repo.count_events(1, event_types=["near"]) == 1

    # Test anchor query
    anchor = seeded_repo.get_anchor_event(1, "object_disappeared", "laptop")
    assert anchor is not None
    assert anchor.t_start == 35.0
