"""Tests for Gemini LLM query parser with mocking (test_parser_llm.py).

All Gemini calls are mocked so no real API key or internet access is needed.
"""

from unittest.mock import MagicMock, patch
import pytest

from cctv.query.parser_llm import _QUERY_CACHE, parse_query, parse_query_gemini
from cctv.query.schema import QuerySpec


@pytest.fixture(autouse=True)
def clear_query_cache():
    _QUERY_CACHE.clear()
    yield
    _QUERY_CACHE.clear()


def test_valid_json_returns_query_spec(monkeypatch):
    """Mock Gemini returning valid QuerySpec structured response."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_test_key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = """{
        "intent": "who_near",
        "event_types": ["near"],
        "classes": ["person", "laptop"],
        "track_id": null,
        "time_range": null,
        "temporal_relation": {
            "relation": "before",
            "anchor": {"event_type": "object_disappeared", "class_name": "laptop"},
            "window_s": 30.0
        },
        "limit": 20,
        "raw_question": "was someone near the computer before it was removed?"
    }"""
    mock_client.models.generate_content.return_value = mock_response

    spec = parse_query_gemini(
        "was someone near the computer before it was removed?",
        client=mock_client,
    )

    assert spec.intent == "who_near"
    assert "laptop" in spec.classes
    assert spec.temporal_relation is not None
    assert spec.temporal_relation.anchor.event_type == "object_disappeared"
    assert spec.parser_used == "gemini"


def test_invalid_json_falls_back_to_rules(monkeypatch):
    """Mock Gemini returning malformed JSON -> should fall back gracefully to rule parser."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_test_key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = "This is not valid JSON at all!"
    mock_client.models.generate_content.return_value = mock_response

    # Using parse_query in 'gemini' mode
    spec = parse_query(
        "How many people are there?",
        mode="gemini",
        client=mock_client,
    )

    # Should have fallen back to rules
    assert "rules" in spec.parser_used
    assert spec.intent == "count"
    assert "person" in spec.classes


def test_unknown_event_type_rejected(monkeypatch):
    """Mock Gemini returning hallucinated / non-existent event types -> rejected."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_test_key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = """{
        "intent": "list_events",
        "event_types": ["person_flying", "car_exploding"],
        "classes": ["person", "ufo"],
        "track_id": null,
        "time_range": null,
        "temporal_relation": null,
        "limit": 20,
        "raw_question": "did any UFO appear?"
    }"""
    mock_client.models.generate_content.return_value = mock_response

    spec = parse_query_gemini(
        "did any UFO appear?",
        client=mock_client,
    )

    # Invalid event types and classes filtered out
    assert "person_flying" not in spec.event_types
    assert "ufo" not in spec.classes


def test_timeout_falls_back_to_rules(monkeypatch):
    """Mock Gemini timeout exception -> falls back gracefully."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_test_key")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("Gemini API timed out after 10s")

    spec = parse_query(
        "Show everything about person 3",
        mode="gemini",
        client=mock_client,
    )

    assert "rules" in spec.parser_used
    assert spec.track_id == 3


def test_missing_key_falls_back_to_rules(monkeypatch):
    """When GEMINI_API_KEY is not set -> falls back to rules with notice."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    spec = parse_query(
        "When did the first person enter?",
        mode="gemini",
    )

    assert "rules" in spec.parser_used
    assert spec.intent == "first"


def test_cache_hit_prevents_second_api_call(monkeypatch):
    """Repeated question should hit in-memory cache and avoid calling the model API again."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_test_key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = """{
        "intent": "what_happened",
        "event_types": [],
        "classes": [],
        "track_id": null,
        "time_range": null,
        "temporal_relation": null,
        "limit": 20,
        "raw_question": "summarize the video"
    }"""
    mock_client.models.generate_content.return_value = mock_response

    q = "summarize the video"
    spec1 = parse_query(q, mode="gemini", client=mock_client)
    assert mock_client.models.generate_content.call_count == 1

    # Second call with same question
    spec2 = parse_query(q, mode="gemini", client=mock_client)
    assert mock_client.models.generate_content.call_count == 1  # No additional API call!
    assert spec1 == spec2
