"""Answer generator module: builds factual, template-based summaries and evidence lists."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from cctv.db.repository import EventRecord
from cctv.evidence import format_seconds_to_time
from cctv.query.executor import QueryResult


@dataclass
class EvidenceItem:
    event_id: Optional[int]
    type: str
    t_start: float
    t_end: float
    track_id: int
    object_track_id: Optional[int]
    class_name: str
    object_class: Optional[str]
    confidence: float
    keyframe_path: Optional[str]
    clip_path: Optional[str]
    details: Dict[str, Any]


@dataclass
class Answer:
    summary: str
    evidence: List[EvidenceItem]
    query_spec: Dict[str, Any]
    parser_used: str
    has_results: bool


def _format_event_time_range(t_start: float, t_end: float) -> str:
    """Format time range as MM:SS or MM:SS - MM:SS."""
    if abs(t_start - t_end) < 0.1:
        return format_seconds_to_time(t_start)
    return f"{format_seconds_to_time(t_start)} - {format_seconds_to_time(t_end)}"


def build_answer(query_result: QueryResult) -> Answer:
    """Produces a factual, deterministic natural language answer and evidence items."""
    spec = query_result.spec
    events = query_result.events
    parser_used = spec.parser_used or "rules"

    # Convert events into EvidenceItems
    evidence_items: List[EvidenceItem] = []
    for ev in events:
        import json
        details = {}
        if ev.details_json:
            try:
                details = json.loads(ev.details_json)
            except Exception:
                pass

        evidence_items.append(
            EvidenceItem(
                event_id=ev.id,
                type=ev.type,
                t_start=ev.t_start,
                t_end=ev.t_end,
                track_id=ev.track_id,
                object_track_id=ev.object_track_id,
                class_name=ev.class_name,
                object_class=ev.object_class,
                confidence=ev.confidence,
                keyframe_path=ev.keyframe_path,
                clip_path=ev.clip_path,
                details=details,
            )
        )

    # Handle Unknown Intent
    if spec.intent == "unknown":
        return Answer(
            summary=(
                "I could not understand that question based on the CCTV event capabilities. "
                "You can ask questions like: 'How many people are there?', "
                "'When did the first person enter?', 'Who was near the laptop before it disappeared?', "
                "'What happened between 0:10 and 0:30?', or 'Show everything about person 2'."
            ),
            evidence=[],
            query_spec=spec.model_dump(),
            parser_used=parser_used,
            has_results=False,
        )

    # Handle Count Intent
    if spec.intent == "count":
        cnt = query_result.total_count if query_result.total_count is not None else len(events)
        color_prefix = f"{spec.color} " if spec.color else ""
        if "person" in spec.classes:
            target = f"{color_prefix}person(s)"
        elif spec.classes:
            target = f"{color_prefix}{spec.classes[0]}(s)"
        else:
            target = f"{color_prefix}event(s)"
        summary = f"Found a total of {cnt} {target} in the video."
        return Answer(
            summary=summary,
            evidence=evidence_items,
            query_spec=spec.model_dump(),
            parser_used=parser_used,
            has_results=cnt > 0,
        )

    # Handle Empty Results
    if not events:
        if spec.color:
            color_target = f"{spec.color} {spec.classes[0] if spec.classes else 'entity'}"
            summary = f"No {color_target} was detected in this video."
        elif spec.temporal_relation and not query_result.anchor_event:
            anchor_name = spec.temporal_relation.anchor.class_name or "object"
            ev_name = spec.temporal_relation.anchor.event_type.replace("_", " ")
            summary = f"No events found: the referenced anchor event ('{anchor_name} {ev_name}') did not occur in this video."
        else:
            summary = "No matching events were found in the database for your query."

        return Answer(
            summary=summary,
            evidence=[],
            query_spec=spec.model_dump(),
            parser_used=parser_used,
            has_results=False,
        )

    # Handle "who_near" intent
    if spec.intent == "who_near":
        # Group near intervals by person track
        people_near = {}
        for ev in events:
            pid = ev.track_id
            t_str = f"({_format_event_time_range(ev.t_start, ev.t_end)})"
            if pid not in people_near:
                people_near[pid] = []
            people_near[pid].append(t_str)

        person_strs = [
            f"Person {pid} {' and '.join(times)}" for pid, times in people_near.items()
        ]
        obj_target = spec.classes[1] if len(spec.classes) > 1 else "the object"

        if query_result.anchor_event:
            anchor = query_result.anchor_event
            anchor_time = format_seconds_to_time(anchor.t_start)
            summary = (
                f"{len(people_near)} person(s) were near {obj_target} in the {int(spec.temporal_relation.window_s)}s "
                f"before it {anchor.type.replace('_', ' ')} at {anchor_time}: {', '.join(person_strs)}."
            )
        else:
            summary = f"{len(people_near)} person(s) were near {obj_target}: {', '.join(person_strs)}."

        return Answer(
            summary=summary,
            evidence=evidence_items,
            query_spec=spec.model_dump(),
            parser_used=parser_used,
            has_results=True,
        )

    # Handle "first" intent
    if spec.intent == "first":
        ev = events[0]
        time_str = format_seconds_to_time(ev.t_start)
        summary = f"The first {ev.class_name} {ev.type.replace('_', ' ')} (ID {ev.track_id}) was recorded at {time_str}."
        return Answer(
            summary=summary,
            evidence=evidence_items,
            query_spec=spec.model_dump(),
            parser_used=parser_used,
            has_results=True,
        )

    # Handle "last" intent
    if spec.intent == "last":
        ev = events[0]
        time_str = format_seconds_to_time(ev.t_start)
        summary = f"The last recorded {ev.class_name} {ev.type.replace('_', ' ')} (ID {ev.track_id}) was at {time_str}."
        return Answer(
            summary=summary,
            evidence=evidence_items,
            query_spec=spec.model_dump(),
            parser_used=parser_used,
            has_results=True,
        )

    # Handle "what_happened" / "list_events"
    ev_summaries = []
    for ev in events[:5]:
        time_str = _format_event_time_range(ev.t_start, ev.t_end)
        ev_desc = f"{ev.type.replace('_', ' ')} ({ev.class_name} #{ev.track_id}) at {time_str}"
        ev_summaries.append(ev_desc)

    summary = f"Retrieved {len(events)} event(s): " + "; ".join(ev_summaries)
    if len(events) > 5:
        summary += f"; and {len(events) - 5} more."

    return Answer(
        summary=summary,
        evidence=evidence_items,
        query_spec=spec.model_dump(),
        parser_used=parser_used,
        has_results=True,
    )
