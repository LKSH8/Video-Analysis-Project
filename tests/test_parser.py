"""Tests for deterministic rule-based query parser (test_parser.py).

Contains at least 25 distinct English test cases covering entities, synonyms,
verbs, time expressions, temporal relations, and gibberish queries.
"""

import pytest
from cctv.query.parser_rules import parse_query_rules


def test_entities_and_synonyms():
    # 1. laptop synonym "computer"
    s1 = parse_query_rules("Did anyone touch the computer?")
    assert "laptop" in s1.classes

    # 2. laptop synonym "notebook"
    s2 = parse_query_rules("When was the notebook placed on the desk?")
    assert "laptop" in s2.classes

    # 3. bag synonym "backpack"
    s3 = parse_query_rules("Who brought the backpack?")
    assert "backpack" in s3.classes

    # 4. phone synonym "mobile"
    s4 = parse_query_rules("Where is the mobile?")
    assert "cell phone" in s4.classes

    # 5. person synonym "guy"
    s5 = parse_query_rules("When did that guy enter?")
    assert "person" in s5.classes
    assert "person_appeared" in s5.event_types

    # 6. bottle
    s6 = parse_query_rules("Show me the bottle")
    assert "bottle" in s6.classes


def test_verbs_and_events():
    # 7. enter -> person_appeared
    s7 = parse_query_rules("When did a person enter the room?")
    assert "person_appeared" in s7.event_types

    # 8. leave -> person_disappeared
    s8 = parse_query_rules("When did someone leave?")
    assert "person_disappeared" in s8.event_types

    # 9. went missing -> object_disappeared
    s9 = parse_query_rules("When did the laptop go missing?")
    assert "object_disappeared" in s9.event_types

    # 10. near / close to
    s10 = parse_query_rules("Who stood close to the chair?")
    assert "near" in s10.event_types or "loitering" in s10.event_types

    # 11. loitering / stay in one place
    s11 = parse_query_rules("Did anyone stay in one place for a long time?")
    assert "loitering" in s11.event_types

    # 12. loiter / linger
    s12 = parse_query_rules("Show all loitering events")
    assert "loitering" in s12.event_types


def test_intents():
    # 13. count people
    s13 = parse_query_rules("How many people are there?")
    assert s13.intent == "count"
    assert "person" in s13.classes

    # 14. first person enter
    s14 = parse_query_rules("When did the first person arrive?")
    assert s14.intent == "first"
    assert "person_appeared" in s14.event_types

    # 15. last seen
    s15 = parse_query_rules("When was person 2 last seen?")
    assert s15.intent == "last"
    assert s15.track_id == 2

    # 16. who near
    s16 = parse_query_rules("Who was near the laptop?")
    assert s16.intent == "who_near"
    assert "laptop" in s16.classes

    # 17. what happened / summarize
    s17 = parse_query_rules("What happened in the video?")
    assert s17.intent == "what_happened"

    # 18. show everything
    s18 = parse_query_rules("Show everything that occurred")
    assert s18.intent == "what_happened"


def test_time_expressions():
    # 19. between seconds
    s19 = parse_query_rules("What happened between 10 and 40 seconds?")
    assert s19.time_range is not None
    assert s19.time_range.start_s == 10.0
    assert s19.time_range.end_s == 40.0

    # 20. between MM:SS
    s20 = parse_query_rules("What happened between 0:10 and 0:30?")
    assert s20.time_range is not None
    assert s20.time_range.start_s == 10.0
    assert s20.time_range.end_s == 30.0

    # 21. in the first 30 seconds
    s21 = parse_query_rules("Who entered in the first 30 seconds?")
    assert s21.time_range is not None
    assert s21.time_range.start_s == 0.0
    assert s21.time_range.end_s == 30.0

    # 22. after 20 seconds
    s22 = parse_query_rules("Did anyone appear after 20 seconds?")
    assert s22.time_range is not None
    assert s22.time_range.start_s == 20.0


def test_temporal_relations():
    # 23. before laptop disappeared
    s23 = parse_query_rules("Who was near the laptop before it disappeared?")
    assert s23.intent == "who_near"
    assert s23.temporal_relation is not None
    assert s23.temporal_relation.relation == "before"
    assert s23.temporal_relation.anchor.event_type == "object_disappeared"
    assert s23.temporal_relation.anchor.class_name == "laptop"

    # 24. after person entered
    s24 = parse_query_rules("What happened after the person entered?")
    assert s24.temporal_relation is not None
    assert s24.temporal_relation.relation == "after"
    assert s24.temporal_relation.anchor.event_type == "person_appeared"


def test_track_ids():
    # 25. explicit person ID
    s25 = parse_query_rules("Show everything about person 3")
    assert s25.track_id == 3
    assert "person" in s25.classes

    # 26. P4 notation
    s26 = parse_query_rules("When did P4 enter?")
    assert s26.track_id == 4


def test_color_queries():
    # 32. red car query
    c1 = parse_query_rules("Did you see any red car?")
    assert c1.color == "red"
    assert "car" in c1.classes

    # 33. black bag query
    c2 = parse_query_rules("Was there any black backpack detected?")
    assert c2.color == "black"
    assert "backpack" in c2.classes

    # 34. blue vehicle count
    c3 = parse_query_rules("How many blue vehicles arrived?")
    assert c3.color == "blue"
    assert c3.intent == "count"
    assert "car" in c3.classes


def test_gibberish_and_out_of_domain():
    # 27. Gibberish 1
    g1 = parse_query_rules("asdfghjk qwertyuiop")
    assert g1.intent == "unknown"

    # 28. Gibberish 2: completely unrelated question
    g2 = parse_query_rules("What is the capital of France?")
    assert g2.intent == "unknown"

    # 29. Gibberish 3: cooking recipe
    g3 = parse_query_rules("How to bake a chocolate cake?")
    assert g3.intent == "unknown"

    # 30. Gibberish 4: nonsense punctuation
    g4 = parse_query_rules("??? !!! ... ---")
    assert g4.intent == "unknown"

    # 31. Empty string
    g5 = parse_query_rules("")
    assert g5.intent == "unknown"
