"""Query specification schema: JSON contract between query parser and executor."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)


class AnchorEvent(BaseModel):
    event_type: str
    class_name: Optional[str] = None


class TemporalRelation(BaseModel):
    relation: Literal["before", "after", "during"]
    anchor: AnchorEvent
    window_s: float = Field(default=30.0, ge=0.0)


class QuerySpec(BaseModel):
    """Pydantic model defining the contract for structured CCTV queries."""
    intent: Literal[
        "list_events",
        "count",
        "first",
        "last",
        "who_near",
        "what_happened",
        "unknown",
    ] = "list_events"
    event_types: List[str] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    color: Optional[str] = None
    track_id: Optional[int] = None
    time_range: Optional[TimeRange] = None
    temporal_relation: Optional[TemporalRelation] = None
    limit: int = Field(default=20, ge=1, le=100)
    raw_question: str
    parser_used: Optional[str] = "rules"  # "rules" | "gemini"
