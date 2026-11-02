"""Rule-based, offline, deterministic English query parser for CCTV events."""

import re
from typing import List, Optional, Tuple
from cctv.query.schema import AnchorEvent, QuerySpec, TemporalRelation, TimeRange

# Synonym mappings
CLASS_SYNONYMS = {
    "laptop": ["laptop", "laptops", "computer", "computers", "notebook", "pc"],
    "backpack": ["backpack", "backpacks", "bag", "bags", "rucksack", "knapsack"],
    "handbag": ["handbag", "handbags", "purse", "purses"],
    "suitcase": ["suitcase", "suitcases", "luggage"],
    "cell phone": ["phone", "phones", "mobile", "cellphone", "cell phone", "smartphone"],
    "bottle": ["bottle", "bottles", "water bottle", "drink"],
    "book": ["book", "books", "textbook", "novel"],
    "cup": ["cup", "cups", "mug", "coffee cup"],
    "chair": ["chair", "chairs", "seat", "seats"],
    "car": ["car", "cars", "vehicle", "vehicles", "automobile", "automobiles", "auto", "sedan", "suv", "cab", "taxi"],
    "person": ["person", "people", "someone", "anyone", "somebody", "anybody", "man", "men", "woman", "women", "guy", "guys", "individual", "individuals"],
}

VERB_TO_EVENT = {
    "person_appeared": ["enter", "entered", "enters", "came in", "come in", "arrived", "arrive", "arriving", "appear", "appeared", "appearing", "detected", "seen", "spotted"],
    "person_disappeared": ["leave", "left", "leaves", "exited", "exit", "exits", "went out", "go out", "departed", "depart"],
    "object_disappeared": ["disappear", "disappeared", "disappearing", "missing", "went missing", "gone", "vanish", "vanished", "stole", "stolen", "steal", "stealing", "take", "took", "taken", "remove", "removed"],
    "object_appeared": ["placed", "brought", "put down"],
    "near": ["near", "close to", "next to", "beside", "by", "around", "approached", "approach", "hanging around", "stand near", "walk near"],
    "loitering": ["loiter", "loitering", "loitered", "stand", "stood", "stay", "stayed", "lingered", "linger", "lingering", "wait", "waited", "waiting", "one place", "one spot", "long time"],
}


COLOR_SYNONYMS = {
    "red": ["red", "crimson", "maroon", "scarlet"],
    "blue": ["blue", "navy", "cyan"],
    "green": ["green", "lime"],
    "yellow": ["yellow", "gold"],
    "white": ["white"],
    "black": ["black", "dark"],
    "gray": ["gray", "grey", "silver"],
    "orange": ["orange"],
    "purple": ["purple", "violet"],
}

def _extract_color(text: str) -> Optional[str]:
    text_lower = text.lower()
    for canonical, syns in COLOR_SYNONYMS.items():
        for syn in syns:
            if re.search(rf"\b{re.escape(syn)}\b", text_lower):
                return canonical
    return None


def _parse_time_str(time_str: str) -> Optional[float]:
    """Parse 'MM:SS' or 'SS' into seconds."""
    time_str = time_str.strip()
    if ":" in time_str:
        parts = time_str.split(":")
        try:
            return float(parts[0]) * 60.0 + float(parts[1])
        except (ValueError, IndexError):
            return None
    try:
        return float(time_str)
    except ValueError:
        return None


def _extract_time_range(text: str) -> Optional[TimeRange]:
    """Extract explicit time bounds from query."""
    # Pattern: 'between X and Y (seconds|s)?' or 'between MM:SS and MM:SS'
    m_between = re.search(r"between\s+([0-9:]+)\s+and\s+([0-9:]+)(?:\s*(?:seconds|secs|s))?", text, re.IGNORECASE)
    if m_between:
        t1 = _parse_time_str(m_between.group(1))
        t2 = _parse_time_str(m_between.group(2))
        if t1 is not None and t2 is not None:
            return TimeRange(start_s=min(t1, t2), end_s=max(t1, t2))

    # Pattern: 'in the first X (seconds|s|minutes)'
    m_first = re.search(r"in\s+the\s+first\s+([0-9]+)\s*(seconds|secs|s|minutes|mins)?", text, re.IGNORECASE)
    if m_first:
        val = float(m_first.group(1))
        unit = (m_first.group(2) or "seconds").lower()
        multiplier = 60.0 if "min" in unit else 1.0
        return TimeRange(start_s=0.0, end_s=val * multiplier)

    # Pattern: 'in the last minute'
    if re.search(r"in\s+the\s+last\s+minute", text, re.IGNORECASE):
        # Time relative to end will be handled if duration known, else placeholder 60s
        return TimeRange(start_s=0.0, end_s=60.0)

    # Pattern: 'after X (seconds|s)?' or 'after MM:SS'
    m_after = re.search(r"after\s+([0-9:]+)(?:\s*(?:seconds|secs|s))?", text, re.IGNORECASE)
    if m_after:
        t = _parse_time_str(m_after.group(1))
        if t is not None:
            return TimeRange(start_s=t, end_s=36000.0)

    # Pattern: 'before X (seconds|s)?' or 'before MM:SS' (only when not followed by event words)
    m_before = re.search(r"before\s+([0-9:]+)(?:\s*(?:seconds|secs|s))?", text, re.IGNORECASE)
    if m_before:
        t = _parse_time_str(m_before.group(1))
        if t is not None:
            return TimeRange(start_s=0.0, end_s=t)

    # Pattern: 'at X (seconds|s)?' or 'at MM:SS'
    m_at = re.search(r"at\s+([0-9:]+)(?:\s*(?:seconds|secs|s))?", text, re.IGNORECASE)
    if m_at:
        t = _parse_time_str(m_at.group(1))
        if t is not None:
            return TimeRange(start_s=max(0.0, t - 1.0), end_s=t + 1.0)

    return None


def _extract_classes(text: str) -> List[str]:
    """Identify referenced object or person classes using synonym sets."""
    found: List[str] = []
    text_lower = text.lower()
    for canonical, syns in CLASS_SYNONYMS.items():
        for syn in syns:
            # Word boundary match
            if re.search(rf"\b{re.escape(syn)}\b", text_lower):
                if canonical not in found:
                    found.append(canonical)
                break
    return found


def _extract_track_id(text: str) -> Optional[int]:
    """Extract explicit track / person ID like 'person 3', 'P3', 'ID 3', '#3'."""
    patterns = [
        r"(?:person|p|id|track|object)\s*#?\s*([0-9]+)",
        r"#([0-9]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def _extract_temporal_relation(text: str, classes: List[str]) -> Optional[TemporalRelation]:
    """Detect before/after anchor clauses like 'before the laptop disappeared'."""
    text_lower = text.lower()
    relation_match = re.search(r"\b(before|after|during)\b\s+(?:the\s+)?([a-z0-9\s]+)", text_lower)
    if not relation_match:
        return None

    rel_type = relation_match.group(1)  # before, after, during
    clause = relation_match.group(2)

    # Identify anchor event type
    anchor_event_type = None
    if any(w in clause for w in ["disappear", "disappeared", "missing", "gone", "stolen", "vanish"]):
        anchor_event_type = "object_disappeared"
    elif any(w in clause for w in ["enter", "entered", "came in", "appear", "arrived"]):
        anchor_event_type = "person_appeared"
    elif any(w in clause for w in ["leave", "left", "exited"]):
        anchor_event_type = "person_disappeared"

    if not anchor_event_type:
        return None

    # Identify anchor class
    anchor_class = None
    for canonical, syns in CLASS_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(s)}\b", clause) for s in syns):
            anchor_class = canonical
            break

    if not anchor_class and anchor_event_type == "object_disappeared":
        # Default to laptop if not stated or if laptop is in general classes
        anchor_class = "laptop" if "laptop" in classes else (classes[0] if classes else "laptop")
    elif not anchor_class and "person" in anchor_event_type:
        anchor_class = "person"

    return TemporalRelation(
        relation=rel_type,  # type: ignore
        anchor=AnchorEvent(event_type=anchor_event_type, class_name=anchor_class),
        window_s=30.0,
    )


def parse_query_rules(question: str) -> QuerySpec:
    """Deterministically parses an English question into a QuerySpec."""
    q = question.strip()
    if not q:
        return QuerySpec(intent="unknown", raw_question=question, parser_used="rules")

    q_lower = q.lower()

    # Track ID check
    track_id = _extract_track_id(q_lower)
    classes = _extract_classes(q_lower)
    color = _extract_color(q_lower)
    time_range = _extract_time_range(q_lower)
    temporal_rel = _extract_temporal_relation(q_lower, classes)

    # Identify matching event types
    matched_events: List[str] = []
    for ev_type, words in VERB_TO_EVENT.items():
        if any(re.search(rf"\b{re.escape(w)}\b", q_lower) for w in words):
            matched_events.append(ev_type)

    # If asking about color existence without an explicit verb ("did you see any red car", "is there a red bag")
    if color and not matched_events:
        if "person" in classes:
            matched_events.append("person_appeared")
        else:
            matched_events.append("object_appeared")

    # Determine intent
    intent = "list_events"

    # Check for gibberish / out of domain queries
    is_cctv_domain = (
        bool(classes)
        or bool(color)
        or bool(matched_events)
        or track_id is not None
        or time_range is not None
        or temporal_rel is not None
        or any(w in q_lower for w in ["what happened", "summarize", "overview", "show everything", "anything happen", "cctv", "camera", "incident", "activity", "see", "saw", "spot", "spotted"])
    )

    if not is_cctv_domain:
        return QuerySpec(intent="unknown", raw_question=question, parser_used="rules")

    # Intent rules
    if re.search(r"\b(how many|count|number of)\b", q_lower):
        intent = "count"
    elif re.search(r"\b(first|earliest|when did.*first)\b", q_lower):
        intent = "first"
    elif re.search(r"\b(last|latest|last seen|when was.*last)\b", q_lower):
        intent = "last"
    elif re.search(r"\b(who was near|who approached|who was close|was anyone near|someone near|was somebody near|anybody near)\b", q_lower):
        intent = "who_near"
        if "near" not in matched_events:
            matched_events.append("near")
    elif any(p in q_lower for p in ["what happened", "summarize", "show everything", "overview", "all events"]):
        intent = "what_happened"

    # Disambiguation for 'appear' (object vs person)
    if "person_appeared" in matched_events and classes and "person" not in classes:
        # Object appeared
        matched_events.remove("person_appeared")
        matched_events.append("object_appeared")

    # Refine classes for 'who_near'
    if intent == "who_near" and not classes:
        classes = ["person", "laptop"]

    # Refine event_types based on intent
    if intent == "what_happened" and not matched_events:
        matched_events = []  # all events

    return QuerySpec(
        intent=intent,  # type: ignore
        event_types=matched_events,
        classes=classes,
        color=color,
        track_id=track_id,
        time_range=time_range,
        temporal_relation=temporal_rel,
        limit=20,
        raw_question=question,
        parser_used="rules",
    )
