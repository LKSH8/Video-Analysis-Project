"""Command line interface for Semantic CCTV system."""

import argparse
import json
import logging
import sys
from pathlib import Path

from cctv.config import get_config
from cctv.db.repository import SQLiteRepository
from cctv.pipeline import process_video
from cctv.query import QueryExecutor, build_answer, parse_query


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_process(args):
    """Run full ingest -> perceive -> events -> evidence pipeline on a video."""
    config = get_config()
    repo = SQLiteRepository(config.system.db_path)

    def print_progress(cur, total, pct, desc="Processing..."):
        sys.stdout.write(f"\r{desc:<70}")
        sys.stdout.flush()

    print(f"=== Processing Video: {args.video_path} ===")
    result = process_video(
        video_path=args.video_path,
        repository=repo,
        start_time=args.start_time,
        config=config,
        progress_callback=print_progress,
    )
    print("\n\nProcessing Complete!")
    print(f"Video ID: {result['video_id']}")
    print(f"Duration: {result['duration_s']:.1f}s")
    print(f"Tracks Extracted: {result['n_tracks']}")
    print(f"Detections Stored: {result['n_detections']}")
    print(f"Events Extracted: {result['n_events']}")
    print(f"Events by Type: {json.dumps(result['events_by_type'], indent=2)}")


def cmd_ask(args):
    """Parse question, execute query against DB, and return factual evidence."""
    config = get_config()
    repo = SQLiteRepository(config.system.db_path)

    video = repo.get_video(args.video_id)
    if not video:
        print(f"Error: Video with ID {args.video_id} not found.")
        sys.exit(1)

    available_classes = repo.get_distinct_classes_in_video(args.video_id)
    available_events = repo.get_distinct_event_types_in_video(args.video_id)

    spec = parse_query(
        question=args.question,
        available_classes=available_classes,
        available_events=available_events,
        video_duration_s=video.duration_s,
        mode=config.query.parser,
    )

    executor = QueryExecutor(repo)
    result = executor.execute(args.video_id, spec)
    answer = build_answer(result)

    print("\n" + "=" * 60)
    print(f"QUESTION: {args.question}")
    print(f"PARSER USED: {answer.parser_used}")
    print("=" * 60)
    print(f"ANSWER: {answer.summary}\n")

    if answer.evidence:
        print(f"EVIDENCE ITEMS ({len(answer.evidence)}):")
        for idx, ev in enumerate(answer.evidence, 1):
            time_str = f"{ev.t_start:.1f}s - {ev.t_end:.1f}s"
            kf_path = ev.keyframe_path or "N/A"
            clip_path = ev.clip_path or "N/A"
            obj_str = f" -> Object #{ev.object_track_id} ({ev.object_class})" if ev.object_track_id else ""
            print(f"  [{idx}] {ev.type.upper()} | {ev.class_name} #{ev.track_id}{obj_str} | Time: {time_str} | Conf: {ev.confidence:.2f}")
            print(f"      Keyframe: {kf_path}")
            print(f"      Clip:     {clip_path}")
    print("=" * 60 + "\n")


def cmd_events(args):
    """List stored events for a video."""
    config = get_config()
    repo = SQLiteRepository(config.system.db_path)

    event_types = [args.type] if args.type else None
    events = repo.query_events(
        video_id=args.video_id,
        event_types=event_types,
        limit=args.limit,
    )

    print(f"\nEvents for Video #{args.video_id} (count: {len(events)}):")
    for ev in events:
        time_str = f"{ev.t_start:.1f}s - {ev.t_end:.1f}s"
        obj_info = f" -> {ev.object_class} #{ev.object_track_id}" if ev.object_track_id else ""
        print(f"- ID {ev.id}: [{ev.type}] {ev.class_name} #{ev.track_id}{obj_info} at {time_str} (conf: {ev.confidence:.2f})")
        if ev.keyframe_path:
            print(f"    Keyframe: {ev.keyframe_path}")


def cmd_stats(args):
    """Show video statistics and event distribution."""
    config = get_config()
    repo = SQLiteRepository(config.system.db_path)

    video = repo.get_video(args.video_id)
    if not video:
        print(f"Error: Video #{args.video_id} not found.")
        sys.exit(1)

    print(f"\n=== Video #{video.id} Statistics ===")
    print(f"Path: {video.path}")
    print(f"Resolution: {video.width}x{video.height} @ {video.fps:.1f} FPS")
    print(f"Duration: {video.duration_s:.1f}s ({video.frame_count} frames)")
    print(f"Status: {video.status}")
    print(f"Processed At: {video.processed_at or 'N/A'}")

    tracks_cnt = repo.count_tracks(video.id)
    people_cnt = repo.count_tracks(video.id, "person")
    print(f"Total Tracks: {tracks_cnt} (People: {people_cnt})")

    event_counts = repo.get_event_counts_by_type(video.id)
    print("\nEvent Distribution:")
    for ev_type, count in event_counts.items():
        print(f"  {ev_type:<20}: {count}")
    print("===============================\n")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Semantic CCTV System CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # process
    p_proc = subparsers.add_parser("process", help="Process a video through the CCTV pipeline")
    p_proc.add_argument("video_path", help="Path to input video file")
    p_proc.add_argument("--start-time", help="Optional wall-clock start time 'YYYY-MM-DD HH:MM:SS'")

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a question about a processed video")
    p_ask.add_argument("question", help="Natural language question")
    p_ask.add_argument("--video-id", type=int, default=1, help="Video ID in database")

    # events
    p_ev = subparsers.add_parser("events", help="List stored events for a video")
    p_ev.add_argument("--video-id", type=int, default=1, help="Video ID in database")
    p_ev.add_argument("--type", help="Filter by event type (e.g. near, person_appeared)")
    p_ev.add_argument("--limit", type=int, default=50, help="Max rows to return")

    # stats
    p_stat = subparsers.add_parser("stats", help="Show video summary statistics")
    p_stat.add_argument("--video-id", type=int, default=1, help="Video ID in database")

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "events":
        cmd_events(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
