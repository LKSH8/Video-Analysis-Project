"""End-to-end pipeline test running on real or synthetic video (test_pipeline_e2e.py)."""

from pathlib import Path
import pytest

from cctv.config import get_config
from cctv.db.repository import SQLiteRepository
from cctv.pipeline import process_video
from cctv.query import QueryExecutor, build_answer, parse_query_rules


def test_pipeline_e2e_sample():
    """Runs full pipeline on data/videos/sample.mp4 if present.
    
    Verifies:
    - tracks > 0
    - events > 0
    - Every event row has keyframe and clip files that actually exist on disk
    - ask 'how many people are there?' returns a non-empty factual answer
    """
    sample_path = Path("data/videos/sample.mp4")
    if not sample_path.exists():
        pytest.skip("Skipping E2E pipeline test: data/videos/sample.mp4 does not exist.")

    config = get_config()
    # Use temporary test database
    test_db_path = "data/test_cctv.db"
    repo = SQLiteRepository(test_db_path)

    result = process_video(
        video_path=str(sample_path),
        repository=repo,
        config=config,
    )

    video_id = result["video_id"]
    assert result["n_tracks"] > 0, "Pipeline should have extracted tracks"
    assert result["n_events"] > 0, "Pipeline should have extracted events"

    # Verify keyframe and clip existence on disk
    events = repo.query_events(video_id=video_id, limit=100)
    for ev in events:
        if ev.keyframe_path:
            assert Path(ev.keyframe_path).exists(), f"Keyframe not found: {ev.keyframe_path}"
        if ev.clip_path:
            assert Path(ev.clip_path).exists(), f"Clip not found: {ev.clip_path}"

    # Verify query
    spec = parse_query_rules("how many people are there?")
    executor = QueryExecutor(repo)
    q_result = executor.execute(video_id, spec)
    answer = build_answer(q_result)

    assert answer.has_results is True
    assert "Found a total of" in answer.summary

    # Clean up test database
    try:
        Path(test_db_path).unlink(missing_ok=True)
    except Exception:
        pass
