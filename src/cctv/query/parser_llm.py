"""Gemini LLM Query Parser with structured outputs and fallback to rules."""

import json
import logging
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from cctv.config import get_config
from cctv.query.parser_rules import parse_query_rules
from cctv.query.schema import QuerySpec

logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Cache for parsed queries: question string -> QuerySpec
_QUERY_CACHE: Dict[str, QuerySpec] = {}

ALLOWED_EVENT_TYPES = {
    "person_appeared": "A person enters the camera field of view",
    "person_disappeared": "A person exits the camera field of view",
    "object_appeared": "An object becomes visible / is placed",
    "object_disappeared": "A previously visible object vanishes or is removed",
    "near": "A person is close to an object for a sustained duration",
    "loitering": "A person stays roughly in the same position for 10+ seconds",
}

ALLOWED_CLASSES = [
    "person",
    "car",
    "laptop",
    "backpack",
    "handbag",
    "suitcase",
    "cell phone",
    "bottle",
    "book",
    "cup",
    "chair",
]

FEW_SHOT_EXAMPLES = [
    {
        "question": "How many people were detected?",
        "spec": {
            "intent": "count",
            "event_types": ["person_appeared"],
            "classes": ["person"],
            "track_id": None,
            "time_range": None,
            "temporal_relation": None,
            "limit": 20,
            "raw_question": "How many people were detected?",
            "parser_used": "gemini",
        },
    },
    {
        "question": "When did the first person enter?",
        "spec": {
            "intent": "first",
            "event_types": ["person_appeared"],
            "classes": ["person"],
            "track_id": None,
            "time_range": None,
            "temporal_relation": None,
            "limit": 1,
            "raw_question": "When did the first person enter?",
            "parser_used": "gemini",
        },
    },
    {
        "question": "Who was near the laptop before it disappeared?",
        "spec": {
            "intent": "who_near",
            "event_types": ["near"],
            "classes": ["person", "laptop"],
            "track_id": None,
            "time_range": None,
            "temporal_relation": {
                "relation": "before",
                "anchor": {"event_type": "object_disappeared", "class_name": "laptop"},
                "window_s": 30.0,
            },
            "limit": 20,
            "raw_question": "Who was near the laptop before it disappeared?",
            "parser_used": "gemini",
        },
    },
    {
        "question": "What happened between 0:10 and 0:30?",
        "spec": {
            "intent": "what_happened",
            "event_types": [],
            "classes": [],
            "track_id": None,
            "time_range": {"start_s": 10.0, "end_s": 30.0},
            "temporal_relation": None,
            "limit": 20,
            "raw_question": "What happened between 0:10 and 0:30?",
            "parser_used": "gemini",
        },
    },
    {
        "question": "Show everything about person 3",
        "spec": {
            "intent": "list_events",
            "event_types": [],
            "classes": ["person"],
            "track_id": 3,
            "time_range": None,
            "temporal_relation": None,
            "limit": 20,
            "raw_question": "Show everything about person 3",
            "parser_used": "gemini",
        },
    },
    {
        "question": "Did you see any red car?",
        "spec": {
            "intent": "list_events",
            "event_types": ["object_appeared"],
            "classes": ["car"],
            "color": "red",
            "track_id": None,
            "time_range": None,
            "temporal_relation": None,
            "limit": 20,
            "raw_question": "Did you see any red car?",
            "parser_used": "gemini",
        },
    },
    {
        "question": "Did anyone stay in one place for a long time?",
        "spec": {
            "intent": "list_events",
            "event_types": ["loitering"],
            "classes": ["person"],
            "track_id": None,
            "time_range": None,
            "temporal_relation": None,
            "limit": 20,
            "raw_question": "Did anyone stay in one place for a long time?",
            "parser_used": "gemini",
        },
    },
]


def _build_system_instruction(
    available_classes: Optional[List[str]] = None,
    available_events: Optional[List[str]] = None,
) -> str:
    """Build system instructions with schema definition, few-shot examples, and video context."""
    classes_to_show = available_classes or ALLOWED_CLASSES
    events_to_show = available_events or list(ALLOWED_EVENT_TYPES.keys())

    events_desc = "\n".join(
        f"- {k}: {v}" for k, v in ALLOWED_EVENT_TYPES.items() if k in events_to_show
    )

    examples_text = json.dumps(FEW_SHOT_EXAMPLES, indent=2)

    return f"""You are a natural language query parser for an indoor Semantic CCTV event system.
Your job is to parse the user's natural language question into a structured QuerySpec JSON object.

Allowed Event Types:
{events_desc}

Allowed Object Classes:
{', '.join(classes_to_show)}

Instructions:
1. Output strictly valid JSON conforming to QuerySpec.
2. If the user specifies a color attribute (e.g. 'red car', 'black bag', 'blue bottle'), populate the 'color' field with the canonical color (red, blue, green, yellow, white, black, gray, orange, purple).
3. If the user asks about something outside of these event types and classes (e.g., facial recognition, text on shirts, weather), set intent to "unknown".
4. Map entities and synonyms accurately (e.g., 'computer' -> 'laptop', 'rucksack' -> 'backpack', 'hanging around' -> 'near' or 'loitering').
4. For relative temporal questions like 'before X disappeared', populate the 'temporal_relation' field.
5. Never invent event types or classes not in the allowed lists.

Few-Shot Examples:
{examples_text}
"""


def parse_query_gemini(
    question: str,
    available_classes: Optional[List[str]] = None,
    available_events: Optional[List[str]] = None,
    client: Optional[genai.Client] = None,
) -> QuerySpec:
    """Uses Gemini API to convert question text into a structured QuerySpec."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    config = get_config()
    model_name = config.llm.model

    if client is None:
        client = genai.Client(api_key=api_key)

    system_instruction = _build_system_instruction(
        available_classes=available_classes, available_events=available_events
    )

    gen_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=QuerySpec,
        temperature=config.llm.temperature,
    )

    # Call Gemini model
    logger.info("Calling Gemini (%s) to parse question: %s", model_name, question)
    response = client.models.generate_content(
        model=model_name,
        contents=question,
        config=gen_config,
    )

    # Parse and validate response
    if hasattr(response, "parsed") and isinstance(response.parsed, QuerySpec):
        spec = response.parsed
    elif response.text:
        spec_dict = json.loads(response.text)
        spec = QuerySpec.model_validate(spec_dict)
    else:
        raise ValueError("Empty or invalid response from Gemini.")

    # Validate classes and event types against allowed sets
    valid_events = [e for e in spec.event_types if e in ALLOWED_EVENT_TYPES]
    valid_classes = [c for c in spec.classes if c in ALLOWED_CLASSES]

    spec.event_types = valid_events
    spec.classes = valid_classes
    spec.raw_question = question
    spec.parser_used = "gemini"

    return spec


def parse_query(
    question: str,
    available_classes: Optional[List[str]] = None,
    available_events: Optional[List[str]] = None,
    video_duration_s: Optional[float] = None,
    mode: Optional[str] = None,
    client: Optional[genai.Client] = None,
) -> QuerySpec:
    """Unified query parser entrypoint respecting config mode (rules | gemini | hybrid).
    
    Includes in-memory caching and automatic fallback to rules on errors.
    """
    q_clean = question.strip()
    if not q_clean:
        return QuerySpec(intent="unknown", raw_question=question, parser_used="rules")

    # Check cache
    cache_key = f"{q_clean.lower()}|{mode}"
    if cache_key in _QUERY_CACHE:
        logger.info("Query cache hit for: '%s'", q_clean)
        return _QUERY_CACHE[cache_key]

    if mode is None:
        mode = get_config().query.parser

    spec: Optional[QuerySpec] = None

    if mode == "rules":
        spec = parse_query_rules(q_clean)

    elif mode == "gemini":
        try:
            spec = parse_query_gemini(
                question=q_clean,
                available_classes=available_classes,
                available_events=available_events,
                client=client,
            )
        except Exception as e:
            logger.warning("Gemini parsing failed (%s), falling back to rule parser.", e)
            spec = parse_query_rules(q_clean)
            spec.parser_used = "rules (fallback)"

    else:  # "hybrid" mode (default)
        # Run rules first
        rule_spec = parse_query_rules(q_clean)
        # If rules successfully parsed a specific intent, use rules
        if rule_spec.intent != "unknown":
            spec = rule_spec
            spec.parser_used = "rules"
        else:
            # Rules could not parse; try Gemini
            try:
                spec = parse_query_gemini(
                    question=q_clean,
                    available_classes=available_classes,
                    available_events=available_events,
                    client=client,
                )
            except Exception as e:
                logger.info("Hybrid Gemini fallback failed (%s); returning rule spec.", e)
                spec = rule_spec
                spec.parser_used = "rules (fallback)"

    # Clamp time range to video duration if provided
    if spec and spec.time_range and video_duration_s is not None:
        spec.time_range.start_s = max(0.0, min(spec.time_range.start_s, video_duration_s))
        spec.time_range.end_s = max(0.0, min(spec.time_range.end_s, video_duration_s))

    _QUERY_CACHE[cache_key] = spec
    return spec
