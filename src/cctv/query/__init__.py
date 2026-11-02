"""Query understanding and execution package."""

from cctv.query.answer import Answer, EvidenceItem, build_answer
from cctv.query.executor import QueryExecutor, QueryResult
from cctv.query.parser_llm import parse_query, parse_query_gemini
from cctv.query.parser_rules import parse_query_rules
from cctv.query.schema import AnchorEvent, QuerySpec, TemporalRelation, TimeRange

__all__ = [
    "AnchorEvent",
    "Answer",
    "EvidenceItem",
    "QueryExecutor",
    "QueryResult",
    "QuerySpec",
    "TemporalRelation",
    "TimeRange",
    "build_answer",
    "parse_query",
    "parse_query_gemini",
    "parse_query_rules",
]
