"""Query executor module: translates QuerySpec into Repository operations."""

from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple

from cctv.db.repository import EventRecord, Repository
from cctv.query.schema import QuerySpec

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    spec: QuerySpec
    events: List[EventRecord]
    total_count: Optional[int] = None
    anchor_event: Optional[EventRecord] = None
    resolved_time_range: Optional[Tuple[float, float]] = None
    debug_notes: str = ""


class QueryExecutor:
    """Executes a QuerySpec contract against the Repository layer."""

    def __init__(self, repository: Repository):
        self.repo = repository

    def execute(self, video_id: int, spec: QuerySpec) -> QueryResult:
        """Executes the query spec against the specified video in the database."""
        if spec.intent == "unknown":
            return QueryResult(
                spec=spec,
                events=[],
                debug_notes="Query intent was 'unknown' or unparseable.",
            )

        t_start: Optional[float] = None
        t_end: Optional[float] = None
        anchor_event: Optional[EventRecord] = None
        notes = []

        # 1. Resolve explicit time range if present
        if spec.time_range:
            t_start = spec.time_range.start_s
            t_end = spec.time_range.end_s
            notes.append(f"Explicit time filter: [{t_start:.1f}s - {t_end:.1f}s]")

        # 2. Resolve temporal anchor relation if present
        if spec.temporal_relation:
            rel = spec.temporal_relation
            anchor_event = self.repo.get_anchor_event(
                video_id=video_id,
                event_type=rel.anchor.event_type,
                class_name=rel.anchor.class_name,
                order="first",
            )
            if not anchor_event:
                notes.append(
                    f"Anchor event '{rel.anchor.event_type}' ({rel.anchor.class_name}) was not found in video."
                )
                return QueryResult(
                    spec=spec,
                    events=[],
                    anchor_event=None,
                    debug_notes="; ".join(notes),
                )

            anchor_t = anchor_event.t_start
            window = rel.window_s
            notes.append(
                f"Resolved anchor {anchor_event.type} at {anchor_t:.1f}s (window: {window:.1f}s)"
            )

            if rel.relation == "before":
                t_start = max(0.0, anchor_t - window)
                t_end = anchor_t
            elif rel.relation == "after":
                t_start = anchor_event.t_end
                t_end = anchor_event.t_end + window
            elif rel.relation == "during":
                t_start = anchor_event.t_start
                t_end = anchor_event.t_end

        # Resolve color filtering if specified
        color_matching_tids = None
        matching_tracks = []
        if spec.color:
            matching_tracks = self.repo.get_tracks(video_id=video_id, color=spec.color)
            if spec.classes:
                matching_tracks = [t for t in matching_tracks if t.class_name in spec.classes]
            color_matching_tids = {t.track_id for t in matching_tracks}
            notes.append(f"Filtered tracks by color '{spec.color}': {len(color_matching_tids)} matches")

        # 3. Handle different intents
        # A. Count Intent
        if spec.intent == "count":
            if color_matching_tids is not None:
                return QueryResult(
                    spec=spec,
                    events=[],
                    total_count=len(color_matching_tids),
                    anchor_event=anchor_event,
                    resolved_time_range=(t_start, t_end) if t_start is not None else None,
                    debug_notes="; ".join(notes),
                )

            # If counting general people or objects without specific event
            if "person" in spec.classes and not spec.event_types:
                total_tracks = self.repo.count_tracks(video_id=video_id, class_name="person")
                notes.append(f"Counted distinct person tracks: {total_tracks}")
                return QueryResult(
                    spec=spec,
                    events=[],
                    total_count=total_tracks,
                    anchor_event=anchor_event,
                    resolved_time_range=(t_start, t_end) if t_start is not None else None,
                    debug_notes="; ".join(notes),
                )

            # Count matching events
            event_types = spec.event_types or None
            classes = spec.classes or None
            count = self.repo.count_events(
                video_id=video_id,
                event_types=event_types,
                classes=classes,
                track_id=spec.track_id,
                t_start=t_start,
                t_end=t_end,
            )
            notes.append(f"Counted matching events: {count}")
            return QueryResult(
                spec=spec,
                events=[],
                total_count=count,
                anchor_event=anchor_event,
                resolved_time_range=(t_start, t_end) if t_start is not None else None,
                debug_notes="; ".join(notes),
            )

        # B. Who Near Intent
        if spec.intent == "who_near":
            events = self.repo.query_events(
                video_id=video_id,
                event_types=["near"],
                classes=spec.classes if spec.classes else None,
                track_id=spec.track_id,
                t_start=t_start,
                t_end=t_end,
                limit=spec.limit,
                order_desc=False,
            )
            notes.append(f"Retrieved {len(events)} 'near' events")
            return QueryResult(
                spec=spec,
                events=events,
                anchor_event=anchor_event,
                resolved_time_range=(t_start, t_end) if t_start is not None else None,
                debug_notes="; ".join(notes),
            )

        # C. First or Last Intent
        order_desc = spec.intent == "last"
        limit = 1 if spec.intent in ("first", "last") else spec.limit

        events = self.repo.query_events(
            video_id=video_id,
            event_types=spec.event_types if spec.event_types else None,
            classes=spec.classes if spec.classes else None,
            track_id=spec.track_id,
            t_start=t_start,
            t_end=t_end,
            limit=limit,
            order_desc=order_desc,
        )

        if color_matching_tids is not None:
            events = [e for e in events if e.track_id in color_matching_tids or (e.object_track_id and e.object_track_id in color_matching_tids)]
            # If no complex event was stored but the colored entity was seen in video
            if not events and color_matching_tids:
                for t in matching_tracks[:spec.limit]:
                    events.append(
                        EventRecord(
                            id=None,
                            video_id=video_id,
                            camera_id="cam1",
                            type="object_appeared" if t.class_name != "person" else "person_appeared",
                            t_start=t.first_t,
                            t_end=t.last_t,
                            track_id=t.track_id,
                            class_name=t.class_name,
                            confidence=t.best_conf,
                        )
                    )

        notes.append(f"Retrieved {len(events)} events (order_desc={order_desc})")
        return QueryResult(
            spec=spec,
            events=events,
            anchor_event=anchor_event,
            resolved_time_range=(t_start, t_end) if t_start is not None else None,
            debug_notes="; ".join(notes),
        )
