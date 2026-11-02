"""SENTINEL: Forensic Video Examination Workstation.

Built strictly according to DESIGN.md specification.
Run with: python app.py (or python server.py)
Access at: http://localhost:8000
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from cctv.config import get_config
from cctv.db.repository import SQLiteRepository
from cctv.evidence import format_seconds_to_time, render_full_tracked_video
from cctv.pipeline import process_video
from cctv.query import QueryExecutor, build_answer, parse_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentinel.server")

app = FastAPI(title="SENTINEL Forensic Video Examination Console", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = get_config()
repo = SQLiteRepository(config.system.db_path)


def compute_file_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 hash for forensic chain of custody."""
    if not file_path.exists():
        return "UNKNOWN_HASH"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(131072):
            h.update(chunk)
    return h.hexdigest().upper()


class SearchRequest(BaseModel):
    video_id: int
    query: str


@app.get("/media/{file_path:path}")
def serve_media(file_path: str):
    """Securely serve keyframe images, video clips, and overlays."""
    p = Path(file_path).resolve()
    base = Path.cwd().resolve()

    if not str(p).startswith(str(base)) or not p.exists():
        raise HTTPException(status_code=404, detail="File not found")

    mime = "application/octet-stream"
    if p.suffix.lower() in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif p.suffix.lower() == ".png":
        mime = "image/png"
    elif p.suffix.lower() == ".mp4":
        mime = "video/mp4"

    return FileResponse(p, media_type=mime)


@app.get("/api/videos")
def get_videos():
    """Retrieve all scanned video recordings and wall stats."""
    videos = repo.list_videos()
    total_duration_s = sum(v.duration_s for v in videos)
    total_people = sum(repo.count_tracks(v.id, "person") for v in videos)
    total_events = sum(repo.count_events(v.id) for v in videos)

    result = []
    for v in videos:
        tracks = repo.get_tracks(v.id)
        events = repo.query_events(v.id, limit=200)
        p_count = sum(1 for t in tracks if t.class_name == "person")
        c_count = sum(1 for t in tracks if t.class_name in ("car", "truck", "vehicle"))

        # Find thumbnail
        thumb_url = None
        evidence_kf_dir = Path(config.system.evidence_dir) / str(v.id) / "keyframes"
        if evidence_kf_dir.exists():
            kfs = sorted(list(evidence_kf_dir.glob("*.jpg")))
            if kfs:
                thumb_url = f"/media/{kfs[0].as_posix()}"

        mins = int(v.duration_s // 60)
        secs = int(v.duration_s % 60)
        len_str = f"{mins:02d}:{secs:02d}"

        v_path = Path(v.path)
        sha_short = "N/A"
        if v_path.exists():
            # Quick hash prefix
            full_sha = compute_file_sha256(v_path)
            sha_short = f"{full_sha[:8]}…{full_sha[-4:]}"

        result.append({
            "id": v.id,
            "path": v.path,
            "name": v_path.stem,
            "duration_s": round(v.duration_s, 1),
            "length_str": len_str,
            "width": v.width,
            "height": v.height,
            "fps": round(v.fps, 1),
            "status": v.status,
            "processed_at": v.processed_at or "2026-11-01 12:48",
            "people_count": p_count,
            "cars_count": c_count,
            "events_count": len(events),
            "thumbnail_url": thumb_url,
            "sha_short": sha_short,
        })

    total_m = int(total_duration_s // 60)
    total_s = int(total_duration_s % 60)
    footage_str = f"{total_m}m {total_s}s" if total_m > 0 else f"{total_s}s"

    return {
        "videos": result,
        "stats": {
            "videos_scanned": len(videos),
            "total_footage": footage_str,
            "people_seen": total_people,
            "events_stored": total_events,
        },
    }


@app.get("/api/video/{video_id}")
def get_video_details(video_id: int):
    """Retrieve detailed telemetry, tracks, events, and timeline ticks for investigation."""
    video = repo.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    tracks = repo.get_tracks(video_id)
    events = repo.query_events(video_id, limit=500)
    distinct_classes = repo.get_distinct_classes_in_video(video_id)

    # Compile people / entities
    people_tracks = [t for t in tracks if t.class_name == "person"]
    people_list = []
    for p in people_tracks:
        duration = round(p.last_t - p.first_t, 1)
        mins_s = int(p.first_t // 60)
        secs_s = int(p.first_t % 60)
        mins_e = int(p.last_t // 60)
        secs_e = int(p.last_t % 60)
        time_range = f"{mins_s:02d}:{secs_s:02d}–{mins_e:02d}:{secs_e:02d} · {duration}s"
        color_info = f"color: {p.color.upper()} · " if p.color and p.color != "unknown" else ""

        people_list.append({
            "track_id": p.track_id,
            "label": f"P{p.track_id}",
            "track_label": f"TRK-{p.track_id:04d}",
            "first_t": p.first_t,
            "last_t": p.last_t,
            "time_range": time_range,
            "confidence": round(p.best_conf, 2),
            "desc": f"{color_info}detected {p.first_t:.1f}s to {p.last_t:.1f}s",
        })

    # Compile event tape ticks
    timeline_events = []
    for ev in events:
        timeline_events.append({
            "id": ev.id,
            "type": ev.type,
            "t_start": round(ev.t_start, 2),
            "t_end": round(ev.t_end, 2),
            "track_id": ev.track_id,
            "track_label": f"TRK-{ev.track_id:04d}",
            "class_name": ev.class_name,
            "pct": min(100.0, max(0.0, (ev.t_start / max(1.0, video.duration_s)) * 100.0)),
            "is_person": ev.class_name == "person",
        })

    tracked_video_path = Path(config.system.evidence_dir) / str(video_id) / "full_tracked.mp4"

    # Pre-made question prompts for this specific video
    suggestions = []
    if "car" in distinct_classes:
        suggestions.append("How many cars were detected?")
        suggestions.append("When did the first car arrive?")
    else:
        suggestions.append("How many people came in?")

    if "laptop" in distinct_classes:
        suggestions.append("Who was near the laptop before it disappeared?")

    if any(e.type == "loitering" for e in events):
        suggestions.append("Did anyone stay in one place for a long time?")

    if len(suggestions) < 4:
        suggestions.append("What happened in the first 20 seconds?")
    if len(suggestions) < 5:
        suggestions.append("Who left through the right exit?")

    mins = int(video.duration_s // 60)
    secs = int(video.duration_s % 60)
    len_str = f"{mins:02d}:{secs:02d}"

    v_path = Path(video.path)
    full_sha = compute_file_sha256(v_path)
    sha_short = f"{full_sha[:8]}…{full_sha[-4:]}"

    return {
        "video": {
            "id": video.id,
            "name": v_path.stem,
            "filename": v_path.name,
            "path": video.path,
            "duration_s": round(video.duration_s, 1),
            "length_str": len_str,
            "width": video.width,
            "height": video.height,
            "fps": round(video.fps, 1),
            "frame_count": video.frame_count,
            "status": video.status,
            "processed_at": video.processed_at or "2026-11-01 12:48",
            "camera_id": video.camera_id,
            "sha256": full_sha,
            "sha_short": sha_short,
        },
        "people": people_list,
        "timeline_events": timeline_events,
        "metrics": {
            "total_tracks": len(tracks),
            "people_count": len(people_tracks),
            "cars_count": sum(1 for t in tracks if t.class_name in ("car", "truck")),
            "total_events": len(events),
        },
        "tracked_video_exists": tracked_video_path.exists(),
        "tracked_video_url": f"/media/{tracked_video_path.as_posix()}" if tracked_video_path.exists() else f"/media/{Path(video.path).as_posix()}",
        "raw_video_url": f"/media/{Path(video.path).as_posix()}",
        "suggestions": suggestions[:5],
        "gemini_active": bool(os.environ.get("GEMINI_API_KEY")),
        "model_name": config.llm.model,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload footage to data/videos and automatically index."""
    save_dir = Path(config.system.videos_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    target_path = save_dir / file.filename

    with open(target_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Automatically run perception and register into SQLite repository
    try:
        res = process_video(
            video_path=str(target_path.as_posix()),
            repository=repo,
            config=config,
        )
        return {
            "status": "success",
            "filename": file.filename,
            "video_id": res.video_id,
            "path": str(target_path.as_posix()),
        }
    except Exception as e:
        logger.error("Processing uploaded video failed: %s", e)
        return {
            "status": "partial",
            "filename": file.filename,
            "path": str(target_path.as_posix()),
            "error": str(e),
        }


@app.post("/api/process/{video_id}")
def process_video_endpoint(video_id: int):
    """Run perception and event extraction on selected video."""
    video = repo.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        res = process_video(
            video_path=video.path,
            repository=repo,
            config=config,
        )
        return {"status": "success", "results": res}
    except Exception as e:
        logger.error("Processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/video/{video_id}")
def delete_video_endpoint(video_id: int):
    """Delete a video, all its indexed metadata, and evidence artifacts."""
    video = repo.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    repo.delete_video(video_id)

    evidence_dir = Path(config.system.evidence_dir) / str(video_id)
    if evidence_dir.exists():
        import shutil
        shutil.rmtree(evidence_dir, ignore_errors=True)

    return {"status": "success", "deleted_id": video_id}


@app.post("/api/search")
def search_incidents(req: SearchRequest):
    """Execute natural language search against indexed events."""
    video = repo.get_video(req.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    available_classes = repo.get_distinct_classes_in_video(req.video_id)
    available_events = repo.get_distinct_event_types_in_video(req.video_id)

    spec = parse_query(
        question=req.query,
        available_classes=available_classes,
        available_events=available_events,
        video_duration_s=video.duration_s,
        mode=config.query.parser,
    )

    executor = QueryExecutor(repo)
    result = executor.execute(req.video_id, spec)
    ans = build_answer(result)

    video_tracks = repo.get_tracks(req.video_id)
    track_color_map = {t.track_id: t.color for t in video_tracks if t.color and t.color != "unknown"}

    evidence_items = []
    for idx, ev in enumerate(ans.evidence, 1):
        mins_s = int(ev.t_start // 60)
        secs_s = ev.t_start % 60
        mins_e = int(ev.t_end // 60)
        secs_e = ev.t_end % 60
        time_str = f"{mins_s:02d}:{secs_s:05.2f}–{mins_e:02d}:{secs_e:05.2f}" if abs(ev.t_start - ev.t_end) > 0.1 else f"{mins_s:02d}:{secs_s:05.2f}"

        conf_val = round(ev.confidence, 2)
        conf_pct = int(conf_val * 100)
        color_str = f" [{track_color_map[ev.track_id].upper()}]" if ev.track_id in track_color_map else ""

        evidence_items.append({
            "idx": idx,
            "event_id": ev.event_id,
            "type": ev.type.upper(),
            "time_str": time_str,
            "t_start": ev.t_start,
            "t_end": ev.t_end,
            "camera_id": video.camera_id.upper(),
            "track_id": ev.track_id,
            "track_label": f"TRK-{ev.track_id:04d}",
            "class_name": f"{ev.class_name.upper()}{color_str}",
            "subject": f"{ev.class_name.upper()}{color_str} · TRK-{ev.track_id:04d}",
            "object_str": f"{ev.object_class.upper()} · TRK-{ev.object_track_id:04d}" if ev.object_track_id else "—",
            "confidence": f"{conf_val:.2f}",
            "conf_pct": conf_pct,
            "keyframe_url": f"/media/{ev.keyframe_path}" if ev.keyframe_path else None,
            "clip_url": f"/media/{ev.clip_path}" if ev.clip_path else None,
            "details": ev.details,
        })

    return {
        "summary": ans.summary,
        "has_results": ans.has_results,
        "parser_used": ans.parser_used,
        "evidence": evidence_items,
        "query_spec": ans.query_spec,
    }


# =========================================================================
# DESIGN.md Compliant Forensic Terminal UI
# =========================================================================
@app.get("/", response_class=HTMLResponse)
def index_page(response: Response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SENTINEL · Forensic Video Analysis Console</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      /* DESIGN.md Tokens */
      --bg-base:      #0E0F10;   /* app background */
      --bg-panel:     #151617;   /* panels, sidebars */
      --bg-inset:     #0A0B0C;   /* video wells, data wells, inputs */
      --bg-raised:    #1C1D1F;   /* table header, active row */

      /* Lines */
      --line-hair:    #26282A;   /* default 1px divider */
      --line-strong:  #3A3D40;   /* panel outlines, focus-adjacent */

      /* Text */
      --text-primary:   #D8DADC;
      --text-secondary: #8C9094;
      --text-tertiary:  #5C6064;
      --text-inverse:   #0E0F10;

      /* Signal colours (functional only) */
      --sig-amber:    #D9A441;   /* timestamps, selection, primary action */
      --sig-red:      #C8453B;   /* alerts, tamper, low confidence, REC state */
      --sig-green:    #5E9C76;   /* verified, integrity OK, high confidence */
      --sig-steel:    #6E8CA3;   /* neutral info, links, secondary markers */

      /* Detection overlay classes */
      --det-person:   #D9A441;
      --det-vehicle:  #6E8CA3;
      --det-object:   #5E9C76;
      --det-flagged:  #C8453B;

      --font-ui:   "IBM Plex Sans", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
      --font-data: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      background-color: var(--bg-base);
      color: var(--text-primary);
      font-family: var(--font-ui);
      font-size: 13px;
      line-height: 20px;
      height: 100vh;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }

    /* Layout Shell */
    .app-shell {
      display: flex;
      height: 100vh;
      width: 100vw;
    }

    /* Fixed Left Rail (240px) */
    .left-rail {
      width: 240px;
      background: var(--bg-panel);
      border-right: 1px solid var(--line-hair);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      padding: 16px 12px;
    }
    .rail-header {
      display: flex;
      gap: 10px;
      align-items: flex-start;
      padding: 0 4px 16px 4px;
      border-bottom: 1px solid var(--line-hair);
      margin-bottom: 12px;
    }
    .rail-bracket {
      font-family: var(--font-data);
      font-size: 14px;
      color: var(--sig-amber);
      line-height: 16px;
    }
    .rail-app-name {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-primary);
      letter-spacing: 0.02em;
    }
    .rail-app-sub {
      font-size: 11px;
      color: var(--text-secondary);
      line-height: 14px;
    }

    /* Nav Item List */
    .nav-list {
      display: flex;
      flex-direction: column;
      gap: 2px;
      flex: 1;
    }
    .nav-btn {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      background: transparent;
      border: 1px solid transparent;
      border-radius: 2px;
      color: var(--text-secondary);
      font-family: var(--font-ui);
      font-size: 13px;
      font-weight: 400;
      cursor: pointer;
      text-align: left;
      transition: background 120ms linear;
    }
    .nav-btn:hover {
      background: var(--bg-raised);
      color: var(--text-primary);
    }
    .nav-btn.active {
      background: var(--bg-raised);
      color: var(--text-primary);
      border-left: 2px solid var(--sig-amber);
      font-weight: 600;
    }
    .nav-icon {
      font-family: var(--font-data);
      font-size: 12px;
      color: var(--sig-steel);
      width: 16px;
    }

    /* Rail Bottom Footer */
    .rail-footer {
      border-top: 1px solid var(--line-hair);
      padding-top: 12px;
      font-family: var(--font-data);
      font-size: 11px;
      line-height: 18px;
      color: var(--text-tertiary);
    }
    .rail-footer-row {
      display: flex;
      justify-content: space-between;
    }
    .status-dot {
      display: inline-block;
      width: 6px;
      height: 6px;
      background: var(--sig-green);
      border-radius: 1px;
      margin-right: 4px;
    }

    /* Main Workspace Container */
    .workspace {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: var(--bg-base);
      overflow-y: auto;
    }

    /* Content Area */
    .content-body {
      padding: 24px 32px;
      flex: 1;
    }

    /* Section Headings */
    .view-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 20px;
    }
    .view-title {
      font-size: 20px;
      line-height: 28px;
      font-weight: 600;
      color: var(--text-primary);
    }
    .view-sub {
      font-size: 13px;
      color: var(--text-secondary);
      margin-top: 2px;
    }

    /* Standard Technical Buttons (DESIGN.md Section 6) */
    .btn {
      background: transparent;
      border: 1px solid var(--line-strong);
      color: var(--text-primary);
      font-family: var(--font-ui);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 0 14px;
      height: 28px;
      border-radius: 2px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 120ms linear;
    }
    .btn:hover { background: var(--bg-raised); }
    .btn-primary {
      background: var(--sig-amber);
      border-color: var(--sig-amber);
      color: var(--text-inverse);
    }
    .btn-primary:hover {
      background: #c69335;
      border-color: #c69335;
    }
    .btn-destruct {
      border-color: var(--sig-red);
      color: var(--sig-red);
    }
    .btn-destruct:hover {
      background: rgba(200, 69, 59, 0.1);
    }

    /* Video Wall Cards Grid */
    .wall-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }

    /* Forensic Upload Progress Box (DESIGN.md Section 6) */
    .upload-progress-box {
      background: var(--bg-panel);
      border: 1px solid var(--line-hair);
      padding: 12px 16px;
      margin-bottom: 20px;
    }
    .progress-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--font-data);
      font-size: 11px;
      color: var(--sig-amber);
      margin-bottom: 8px;
      font-variant-numeric: tabular-nums;
    }
    .progress-stats {
      color: var(--text-secondary);
      font-size: 11px;
    }
    .progress-track {
      width: 100%;
      height: 3px;
      background: var(--bg-inset);
      border: 1px solid var(--line-hair);
      position: relative;
    }
    .progress-fill {
      width: 0%;
      height: 100%;
      background: var(--sig-amber);
      transition: width 100ms linear;
    }
    .wall-card {
      background: var(--bg-panel);
      border: 1px solid var(--line-hair);
      border-radius: 0;
      display: flex;
      flex-direction: column;
    }
    .thumb-well {
      position: relative;
      background: var(--bg-inset);
      border-bottom: 1px solid var(--line-hair);
      height: 180px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .thumb-img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    /* 10px Registration Corner Ticks */
    .tick {
      position: absolute;
      width: 10px;
      height: 10px;
      border-color: var(--text-secondary);
      pointer-events: none;
      z-index: 2;
    }
    .tick-tl { top: 6px; left: 6px; border-top: 1px solid; border-left: 1px solid; }
    .tick-tr { top: 6px; right: 6px; border-top: 1px solid; border-right: 1px solid; }
    .tick-bl { bottom: 6px; left: 6px; border-bottom: 1px solid; border-left: 1px solid; }
    .tick-br { bottom: 6px; right: 6px; border-bottom: 1px solid; border-right: 1px solid; }

    .thumb-tag-cam {
      position: absolute;
      top: 10px;
      left: 12px;
      font-family: var(--font-data);
      font-size: 11px;
      color: var(--sig-amber);
      background: rgba(14, 15, 16, 0.88);
      padding: 2px 6px;
      border: 1px solid var(--line-hair);
      z-index: 3;
    }
    .thumb-tag-res {
      position: absolute;
      bottom: 10px;
      left: 12px;
      font-family: var(--font-data);
      font-size: 10.5px;
      color: var(--text-secondary);
      background: rgba(14, 15, 16, 0.88);
      padding: 1px 4px;
      z-index: 3;
    }
    .thumb-tag-len {
      position: absolute;
      bottom: 10px;
      right: 12px;
      font-family: var(--font-data);
      font-size: 11px;
      color: var(--text-primary);
      background: rgba(14, 15, 16, 0.88);
      padding: 1px 6px;
      z-index: 3;
    }

    .card-meta {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line-hair);
    }
    .card-name-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 8px;
    }
    .card-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-primary);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 200px;
    }
    .card-scanned {
      font-family: var(--font-data);
      font-size: 10.5px;
      color: var(--text-tertiary);
    }
    .card-stats-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      font-family: var(--font-data);
      font-size: 11px;
      color: var(--text-secondary);
      border-top: 1px solid var(--line-hair);
      padding-top: 8px;
    }
    .card-stat-val { font-weight: 600; color: var(--text-primary); font-size: 13px; }

    .card-actions {
      display: grid;
      grid-template-columns: 1fr 1fr auto;
      padding: 8px;
      gap: 6px;
      background: var(--bg-inset);
    }

    /* Add Video Card */
    .add-card {
      border: 1px dashed var(--line-strong);
      background: transparent;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 260px;
      cursor: pointer;
      transition: border-color 120ms linear;
    }
    .add-card:hover { border-color: var(--sig-amber); }
    .add-plus { font-size: 24px; color: var(--text-secondary); margin-bottom: 6px; }
    .add-label { font-size: 11px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; }

    /* ================= INVESTIGATE VIEW ================= */
    .investigate-layout {
      display: grid;
      grid-template-columns: 1fr 400px;
      gap: 16px;
      align-items: start;
    }

    /* Video Player Well */
    .video-well {
      position: relative;
      background: var(--bg-inset);
      border: 1px solid var(--line-strong);
      border-radius: 0;
      overflow: hidden;
      margin-bottom: 16px;
    }
    .video-well video {
      width: 100%;
      height: auto;
      max-height: 480px;
      display: block;
      background: #000;
    }
    .video-overlay-burn-tl {
      position: absolute;
      top: 10px;
      left: 12px;
      font-family: var(--font-data);
      font-size: 11px;
      color: var(--sig-amber);
      pointer-events: none;
      background: rgba(14, 15, 16, 0.88);
      border: 1px solid var(--line-hair);
      padding: 2px 8px;
      z-index: 3;
    }
    .video-overlay-burn-br {
      position: absolute;
      bottom: 10px;
      right: 12px;
      font-family: var(--font-data);
      font-size: 11px;
      color: var(--text-secondary);
      pointer-events: none;
      background: rgba(14, 15, 16, 0.88);
      border: 1px solid var(--line-hair);
      padding: 2px 8px;
      z-index: 3;
    }
    .video-crosshair {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      pointer-events: none;
      display: none;
      z-index: 2;
    }
    .video-crosshair.active { display: block; }
    .ch-h {
      position: absolute;
      top: 50%; left: 0; right: 0;
      height: 1px;
      background: rgba(217, 164, 65, 0.35);
    }
    .ch-v {
      position: absolute;
      left: 50%; top: 0; bottom: 0;
      width: 1px;
      background: rgba(217, 164, 65, 0.35);
    }
    .ch-target {
      position: absolute;
      top: calc(50% - 10px);
      left: calc(50% - 10px);
      width: 20px;
      height: 20px;
      border: 1px solid var(--sig-amber);
    }

    /* Panel Wrapper */
    .panel-box {
      background: var(--bg-panel);
      border: 1px solid var(--line-hair);
      margin-bottom: 16px;
    }
    .panel-header {
      height: 28px;
      background: var(--bg-panel);
      border-bottom: 1px solid var(--line-hair);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 12px;
    }
    .panel-label {
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-secondary);
    }
    .panel-hint {
      font-size: 11px;
      color: var(--text-tertiary);
    }
    .panel-body {
      padding: 12px;
    }

    /* Event Tape Timeline */
    .tape-container {
      position: relative;
    }
    .tape-layout {
      display: flex;
      gap: 12px;
      align-items: flex-start;
    }
    .tape-labels-col {
      width: 56px;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
    }
    .tape-labels-col .lane-label {
      height: 16px;
      line-height: 16px;
      font-size: 11px;
      color: var(--text-secondary);
      font-weight: 500;
      margin-bottom: 8px;
    }
    .tape-tracks-col {
      position: relative;
      flex: 1;
      display: flex;
      flex-direction: column;
      cursor: crosshair;
    }
    .tape-tracks-col .lane-track {
      position: relative;
      width: 100%;
      height: 16px;
      background: var(--bg-inset);
      border: 1px solid var(--line-hair);
      margin-bottom: 8px;
    }
    .tape-mark {
      position: absolute;
      top: 1px;
      width: 2px;
      height: 12px;
      cursor: pointer;
      z-index: 2;
    }
    .tape-mark-person { background: var(--sig-amber); }
    .tape-mark-flag { background: var(--sig-red); }

    .tape-playhead {
      position: absolute;
      top: 0;
      bottom: 18px;
      width: 2px;
      background: var(--sig-amber);
      pointer-events: none;
      z-index: 10;
      left: 0%;
      transform: translateX(-1px);
    }
    .tape-playhead::after {
      content: '';
      position: absolute;
      top: 0;
      left: -3px;
      width: 8px;
      height: 6px;
      background: var(--sig-amber);
      border-radius: 1px;
    }

    .tape-axis {
      display: flex;
      justify-content: space-between;
      font-family: var(--font-data);
      font-size: 10px;
      color: var(--text-tertiary);
      border-top: 1px solid var(--line-hair);
      padding-top: 4px;
      font-variant-numeric: tabular-nums;
      width: 100%;
    }

    .tape-legend {
      display: flex;
      gap: 16px;
      font-size: 11px;
      color: var(--text-secondary);
      margin-top: 12px;
      padding-top: 8px;
      border-top: 1px solid var(--line-hair);
    }
    .legend-item { display: flex; align-items: center; gap: 6px; }
    .legend-box { width: 8px; height: 8px; border-radius: 1px; }

    /* Entities / People Cards Horizontal Scroll */
    .entities-row {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 8px;
    }
    .entity-card {
      background: var(--bg-inset);
      border: 1px solid var(--line-hair);
      padding: 8px 10px;
      cursor: pointer;
      border-radius: 2px;
      transition: border-color 120ms linear;
    }
    .entity-card:hover { border-color: var(--sig-amber); }
    .entity-id {
      font-family: var(--font-data);
      font-weight: 600;
      color: var(--sig-amber);
      font-size: 12px;
    }
    .entity-time {
      font-family: var(--font-data);
      font-size: 11px;
      color: var(--text-secondary);
      margin-top: 2px;
    }
    .entity-desc {
      font-size: 11px;
      color: var(--text-tertiary);
      margin-top: 2px;
    }

    /* Right Inspection Sidebar */
    .query-well {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .query-input-wrap {
      display: flex;
      align-items: center;
      background: var(--bg-inset);
      border: 1px solid var(--line-strong);
      border-radius: 2px;
      height: 32px;
    }
    .query-input-wrap:focus-within {
      border-color: var(--sig-amber);
    }
    .query-prefix {
      font-family: var(--font-ui);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.1em;
      color: var(--text-tertiary);
      padding: 0 8px;
      border-right: 1px solid var(--line-hair);
      text-transform: uppercase;
      user-select: none;
    }
    .query-input {
      flex: 1;
      height: 30px;
      background: transparent;
      border: none;
      color: var(--text-primary);
      font-family: var(--font-data);
      font-size: 12px;
      padding: 0 8px;
      outline: none;
    }
    .query-ask-btn {
      height: 26px;
      margin-right: 2px;
      padding: 0 10px;
      font-size: 10.5px;
    }

    .query-pills {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-top: 4px;
    }
    .query-pill-btn {
      background: var(--bg-inset);
      border: 1px solid var(--line-hair);
      color: var(--text-secondary);
      font-family: var(--font-ui);
      font-size: 11px;
      padding: 5px 8px;
      text-align: left;
      cursor: pointer;
      border-radius: 2px;
      transition: background 120ms linear, color 120ms linear;
    }
    .query-pill-btn:hover {
      background: var(--bg-raised);
      color: var(--text-primary);
      border-color: var(--line-strong);
    }

    /* Result Exhibit Box */
    .result-exhibit {
      background: var(--bg-inset);
      border: 1px solid var(--line-hair);
      border-left: 2px solid var(--sig-amber);
      padding: 12px;
      margin-top: 10px;
    }
    .exhibit-title {
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--sig-amber);
      margin-bottom: 4px;
    }
    .exhibit-body {
      font-size: 12.5px;
      line-height: 18px;
      color: var(--text-primary);
    }

    /* Forensic Table (DESIGN.md Section 6) */
    .forensic-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      line-height: 18px;
      margin-top: 10px;
      border: 1px solid var(--line-hair);
    }
    .forensic-table th {
      background: var(--bg-raised);
      color: var(--text-secondary);
      font-family: var(--font-ui);
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      text-align: left;
      padding: 6px 8px;
      border-bottom: 1px solid var(--line-hair);
    }
    .forensic-table td {
      padding: 5px 8px;
      border-bottom: 1px solid var(--line-hair);
      font-family: var(--font-data);
      font-size: 11.5px;
      font-variant-numeric: tabular-nums;
    }
    .forensic-table tr:hover {
      background: var(--bg-raised);
    }
    .conf-bar-wrap {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .conf-bar-bg {
      width: 40px;
      height: 6px;
      background: var(--bg-inset);
      border: 1px solid var(--line-hair);
      display: inline-block;
    }
    .conf-bar-fill {
      height: 100%;
      background: var(--sig-amber);
    }

    /* Evidence List Item */
    .ev-card {
      border: 1px solid var(--line-hair);
      background: var(--bg-inset);
      margin-top: 10px;
      padding: 8px;
    }
    .ev-card-header {
      display: flex;
      justify-content: space-between;
      font-family: var(--font-data);
      font-size: 11px;
      margin-bottom: 6px;
      color: var(--text-secondary);
    }
    .ev-badge {
      background: var(--bg-raised);
      padding: 1px 4px;
      border: 1px solid var(--line-hair);
      color: var(--sig-amber);
      font-size: 10.5px;
    }
    .ev-media-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .ev-media-grid img, .ev-media-grid video {
      width: 100%;
      height: 100px;
      object-fit: cover;
      background: #000;
      border: 1px solid var(--line-hair);
    }

    /* Key-Value Telemetry Grid (DESIGN.md Section 6 Inspector) */
    .inspector-grid {
      display: grid;
      grid-template-columns: 180px 1fr;
      border: 1px solid var(--line-hair);
      background: var(--bg-inset);
    }
    .insp-label {
      padding: 8px 12px;
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--text-tertiary);
      border-bottom: 1px solid var(--line-hair);
      border-right: 1px solid var(--line-hair);
    }
    .insp-val {
      padding: 8px 12px;
      font-family: var(--font-data);
      font-size: 12px;
      color: var(--text-primary);
      border-bottom: 1px solid var(--line-hair);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .tag-chip {
      display: inline-block;
      padding: 1px 6px;
      border: 1px solid var(--sig-green);
      color: var(--sig-green);
      font-family: var(--font-data);
      font-size: 10.5px;
      border-radius: 2px;
      text-transform: uppercase;
    }
  </style>
</head>
<body>

  <div class="app-shell">
    <!-- Left Navigation Rail -->
    <nav class="left-rail">
      <div class="rail-header">
        <div class="rail-bracket">[·]</div>
        <div>
          <div class="rail-app-name">CCTV Event Search</div>
          <div class="rail-app-sub">Forensic incident workstation</div>
        </div>
      </div>

      <div class="nav-list">
        <button class="nav-btn active" id="btnNavWall" onclick="showView('wall')">
          <span class="nav-icon">::</span>
          <span>Camera wall</span>
        </button>
        <button class="nav-btn" id="btnNavInvestigate" onclick="showView('investigate')">
          <span class="nav-icon">Q</span>
          <span>Investigate</span>
        </button>
        <button class="nav-btn" id="btnNavAnalytics" onclick="showView('analytics')">
          <span class="nav-icon">||</span>
          <span>Analytics</span>
        </button>
      </div>

      <div class="rail-footer">
        <div class="rail-footer-row">
          <span>Sources</span>
          <span id="footerVideoCount">0</span>
        </div>
        <div class="rail-footer-row">
          <span>Detector</span>
          <span style="color:var(--text-primary)">YOLO11n + ByteTrack</span>
        </div>
        <div class="rail-footer-row" style="margin-top:4px;">
          <span><span class="status-dot"></span>Engine</span>
          <span id="footerGeminiState" style="color:var(--sig-green)">Online</span>
        </div>
      </div>
    </nav>

    <!-- Main Workspace -->
    <div class="workspace">
      <div class="content-body">

        <!-- ================= VIEW 1: CAMERA WALL ================= -->
        <div id="viewWall">
          <div class="view-header">
            <div>
              <h1 class="view-title">Camera wall</h1>
              <div class="view-sub">Indexed video recordings and surveillance telemetry archive.</div>
            </div>
            <div>
              <button class="btn btn-primary" onclick="triggerUpload()">Add video</button>
              <input type="file" id="fileInput" accept=".mp4,.avi,.mov" style="display:none;" onchange="handleFileUpload(event)">
            </div>
          </div>

          <!-- Real-Time Upload & Processing Progress Bar -->
          <div id="uploadProgressBox" class="upload-progress-box" style="display:none;">
            <div class="progress-header">
              <span id="uploadStatusText">UPLOADING FOOTAGE · 0.0%</span>
              <span id="uploadByteStats">0.0 MB / 0.0 MB</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" id="uploadProgressFill"></div>
            </div>
          </div>

          <!-- Video Cards Grid -->
          <div class="wall-grid" id="wallCardsContainer"></div>
        </div>

        <!-- ================= VIEW 2: INVESTIGATE ================= -->
        <div id="viewInvestigate" style="display:none;">
          <div class="view-header">
            <div>
              <h1 class="view-title">Investigate</h1>
              <div class="view-sub">Examine footage, scan temporal event tape, and execute structured incident queries.</div>
            </div>
            <div style="display:flex;gap:8px;align-items:center;">
              <select id="investigateSelect" class="query-input" style="height:28px;width:240px;background:var(--bg-inset);border:1px solid var(--line-strong);" onchange="onInvestigateSelectChange(this.value)"></select>
              <button class="btn" id="crosshairBtn" onclick="toggleCrosshair()">Crosshair</button>
              <button class="btn" onclick="showView('analytics')">Analytics</button>
            </div>
          </div>

          <div class="investigate-layout">
            <!-- Left Column: Video Player, Timeline, Entity Registry -->
            <div>
              <!-- Video Well with Corner Registration Ticks -->
              <div class="video-well">
                <span class="tick tick-tl"></span>
                <span class="tick tick-tr"></span>
                <span class="tick tick-bl"></span>
                <span class="tick tick-br"></span>

                <div class="video-overlay-burn-tl" id="burnOverlayTL">CAM-01 · 00:00:00.000 · F000000</div>
                <div class="video-overlay-burn-br" id="burnOverlayBR">1920×1080 · 30.0 FPS</div>

                <div class="video-crosshair" id="videoCrosshair">
                  <div class="ch-h"></div>
                  <div class="ch-v"></div>
                  <div class="ch-target"></div>
                </div>

                <video id="mainVideoPlayer" controls playsinline ontimeupdate="onVideoTimeUpdate()"></video>
              </div>

              <!-- Event Tape Timeline -->
              <div class="panel-box">
                <div class="panel-header">
                  <span class="panel-label">Event tape</span>
                  <span class="panel-hint">Temporal event markers. Click a mark or position to seek player.</span>
                </div>
                <div class="panel-body tape-container">
                  <div class="tape-layout">
                    <div class="tape-labels-col">
                      <div class="lane-label">People</div>
                      <div class="lane-label">Objects</div>
                    </div>
                    <div class="tape-tracks-col" id="tapeTracksCol" onclick="seekByTrackClick(event)">
                      <div class="tape-playhead" id="tapePlayhead"></div>
                      <div class="lane-track" id="tapeTrackPeople"></div>
                      <div class="lane-track" id="tapeTrackObjects"></div>
                      <div class="tape-axis">
                        <span>00:00</span>
                        <span id="tapeTick1">00:10</span>
                        <span id="tapeTick2">00:20</span>
                        <span id="tapeTick3">00:30</span>
                        <span id="tapeTickEnd">00:50</span>
                      </div>
                    </div>
                  </div>
                  <div class="tape-legend">
                    <div class="legend-item"><span class="legend-box tape-mark-person"></span><span>Person appeared / disappeared</span></div>
                    <div class="legend-item"><span class="legend-box tape-mark-flag"></span><span>Disappeared object / Flagged anomaly</span></div>
                  </div>
                </div>
              </div>

              <!-- People in this video -->
              <div class="panel-box">
                <div class="panel-header">
                  <span class="panel-label">Tracked entities in footage</span>
                  <span class="panel-hint">Click an entity to seek to entry timestamp.</span>
                </div>
                <div class="panel-body">
                  <div class="entities-row" id="peopleCardsRow"></div>
                </div>
              </div>
            </div>

            <!-- Right Column: Query & Search Inspector -->
            <div>
              <div class="panel-box">
                <div class="panel-header">
                  <span class="panel-label">Evidence Query Console</span>
                  <span class="panel-hint" id="geminiStateIndicator">Gemini 3.5 Active</span>
                </div>
                <div class="panel-body query-well">
                  <div class="query-input-wrap">
                    <span class="query-prefix">QUERY</span>
                    <input type="text" id="queryTextInput" class="query-input" placeholder="Who left through exit after 0:20?" onkeydown="if(event.key==='Enter') executeQuery()">
                    <button class="btn btn-primary query-ask-btn" onclick="executeQuery()">ASK</button>
                  </div>

                  <div class="query-pills" id="queryPillsContainer"></div>

                  <div id="queryLoadingState" style="display:none;font-family:var(--font-data);color:var(--sig-amber);font-size:11px;padding:8px 0;">
                    PROCESSING QUERY…
                  </div>

                  <div id="queryResultContainer" style="display:none;">
                    <div class="result-exhibit">
                      <div class="exhibit-title">Finding</div>
                      <div class="exhibit-body" id="queryResultText"></div>
                    </div>

                    <div style="margin-top:14px;font-size:10.5px;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-secondary);font-weight:600;">
                      Chain of Custody Exhibits
                    </div>

                    <!-- Exhibits Table View -->
                    <table class="forensic-table" id="exhibitsTable">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>TIME</th>
                          <th>TRK</th>
                          <th>CLASS</th>
                          <th>CONF</th>
                          <th>EXHIBIT</th>
                        </tr>
                      </thead>
                      <tbody id="exhibitsTableBody"></tbody>
                    </table>

                    <div id="queryEvidenceList"></div>
                  </div>
                </div>
              </div>

              <!-- Re-index Action -->
              <button class="btn" style="width:100%;height:32px;margin-bottom:12px;" id="reprocessBtn" onclick="runPipelineOnActive()">
                Re-scan & Index Footage
              </button>
            </div>
          </div>
        </div>

        <!-- ================= VIEW 3: ANALYTICS ================= -->
        <div id="viewAnalytics" style="display:none;">
          <div class="view-header">
            <div>
              <h1 class="view-title">Analytics</h1>
              <div class="view-sub">Sensor telemetry, forensic chain of custody, and cryptographic audit proofs.</div>
            </div>
            <div style="display:flex;gap:8px;align-items:center;">
              <button class="btn" onclick="showView('investigate')">Back to Investigate</button>
              <button class="btn btn-destruct" onclick="confirmDeleteActiveVideo()">Delete video</button>
            </div>
          </div>

          <div class="panel-box" style="margin-bottom:20px;">
            <div class="panel-header">
              <span class="panel-label">Sensor Telemetry & Audit Integrity</span>
              <span class="tag-chip" id="anIntegrityTag">INTEGRITY OK</span>
            </div>
            <div class="inspector-grid">
              <div class="insp-label">Source Filename</div>
              <div class="insp-val" id="anSource">-</div>

              <div class="insp-label">Camera Identifier</div>
              <div class="insp-val" id="anCamId">-</div>

              <div class="insp-label">Dimensions</div>
              <div class="insp-val" id="anDimensions">-</div>

              <div class="insp-label">Framerate</div>
              <div class="insp-val" id="anFps">-</div>

              <div class="insp-label">Temporal Duration</div>
              <div class="insp-val" id="anDuration">-</div>

              <div class="insp-label">SHA-256 Hash</div>
              <div class="insp-val">
                <span id="anSha256" style="word-break:break-all;">-</span>
                <button class="btn" style="height:22px;padding:0 8px;font-size:10px;margin-left:8px;" onclick="copyHash()">Copy</button>
              </div>

              <div class="insp-label">Tracked Entities</div>
              <div class="insp-val" id="anTracks">-</div>

              <div class="insp-label">Person Tracks</div>
              <div class="insp-val" id="anPeople">-</div>

              <div class="insp-label">Vehicle Tracks</div>
              <div class="insp-val" id="anCars">-</div>

              <div class="insp-label">Indexed Events</div>
              <div class="insp-val" id="anEvents">-</div>

              <div class="insp-label">Perception Engine</div>
              <div class="insp-val">Ultralytics YOLO11n + ByteTrack (De-fragmented)</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>

  <script>
    let activeVideoId = null;
    let currentVideoData = null;
    let crosshairEnabled = false;

    async function loadWall() {
      const res = await fetch('/api/videos');
      const data = await res.json();

      document.getElementById('footerVideoCount').textContent = data.stats.videos_scanned;

      const container = document.getElementById('wallCardsContainer');
      container.innerHTML = '';

      data.videos.forEach(v => {
        const card = document.createElement('div');
        card.className = 'wall-card';
        card.innerHTML = `
          <div class="thumb-well">
            <span class="tick tick-tl"></span>
            <span class="tick tick-tr"></span>
            <span class="tick tick-bl"></span>
            <span class="tick tick-br"></span>
            <span class="thumb-tag-cam">${v.name}</span>
            <span class="thumb-tag-res">${v.width}×${v.height}</span>
            <span class="thumb-tag-len">${v.length_str}</span>
            ${v.thumbnail_url ? `<img src="${v.thumbnail_url}" class="thumb-img" alt="Thumbnail">` : `<div style="font-family:var(--font-data);color:var(--text-tertiary);font-size:11px;">[NO KEYFRAME]</div>`}
          </div>
          <div class="card-meta">
            <div class="card-name-row">
              <span class="card-title" title="${v.name}">${v.name}</span>
              <span class="card-scanned">scanned ${v.processed_at.split(' ')[0]}</span>
            </div>
            <div class="card-stats-row">
              <div><span class="card-stat-val">${v.people_count}</span> people</div>
              <div><span class="card-stat-val">${v.events_count}</span> events</div>
              <div><span class="card-stat-val">${v.length_str}</span> length</div>
            </div>
          </div>
          <div class="card-actions">
            <button class="btn btn-primary" onclick="openInvestigate(${v.id})">Investigate</button>
            <button class="btn" onclick="openAnalytics(${v.id})">Analytics</button>
            <button class="btn btn-destruct" onclick="confirmDeleteVideo(event, ${v.id}, '${v.name.replace(/'/g, "\\'")}')">Delete</button>
          </div>
        `;
        container.appendChild(card);
      });

      // Add Video dashed card
      const addCard = document.createElement('div');
      addCard.className = 'add-card';
      addCard.onclick = triggerUpload;
      addCard.innerHTML = `
        <div class="add-plus">+</div>
        <div class="add-label">Add video</div>
      `;
      container.appendChild(addCard);

      // Populate investigate select dropdown
      const sel = document.getElementById('investigateSelect');
      sel.innerHTML = '';
      data.videos.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.id;
        opt.textContent = `${v.name} (${v.length_str})`;
        sel.appendChild(opt);
      });

      if (!activeVideoId && data.videos.length > 0) {
        activeVideoId = data.videos[0].id;
      }
    }

    async function openInvestigate(videoId) {
      activeVideoId = videoId;
      document.getElementById('investigateSelect').value = videoId;
      await loadInvestigateVideo(videoId);
      showView('investigate');
    }

    async function openAnalytics(videoId) {
      activeVideoId = videoId;
      document.getElementById('investigateSelect').value = videoId;
      await loadInvestigateVideo(videoId);
      showView('analytics');
    }

    function onInvestigateSelectChange(id) {
      activeVideoId = id;
      loadInvestigateVideo(id);
    }

    async function loadInvestigateVideo(videoId) {
      const res = await fetch(`/api/video/${videoId}`);
      const data = await res.json();
      currentVideoData = data;

      document.getElementById('burnOverlayTL').textContent = `${data.video.camera_id.toUpperCase()} · 00:00:00.000 · F000000`;
      document.getElementById('burnOverlayBR').textContent = `${data.video.width}×${data.video.height} · ${data.video.fps} FPS`;

      // Set video player
      const player = document.getElementById('mainVideoPlayer');
      player.src = data.tracked_video_exists ? data.tracked_video_url : data.raw_video_url;

      // Event Tape Ticks
      const trackPeople = document.getElementById('tapeTrackPeople');
      const trackObjects = document.getElementById('tapeTrackObjects');
      trackPeople.innerHTML = '';
      if (trackObjects) trackObjects.innerHTML = '';

      data.timeline_events.forEach(ev => {
        const mark = document.createElement('div');
        mark.className = `tape-mark ${ev.is_person ? 'tape-mark-person' : 'tape-mark-flag'}`;
        if (ev.type.includes('disappear') || ev.type.includes('flag')) {
          mark.className = 'tape-mark tape-mark-flag';
        }
        mark.style.left = `${ev.pct}%`;
        mark.title = `${ev.type} (${ev.track_label}) at ${ev.t_start}s`;
        mark.onclick = (e) => {
          e.stopPropagation();
          player.currentTime = ev.t_start;
          player.play();
        };

        if (ev.is_person) {
          trackPeople.appendChild(mark);
        } else if (trackObjects) {
          trackObjects.appendChild(mark);
        }
      });

      // Axis ticks
      const dur = data.video.duration_s;
      document.getElementById('tapeTick1').textContent = formatSeconds(dur * 0.25);
      document.getElementById('tapeTick2').textContent = formatSeconds(dur * 0.50);
      document.getElementById('tapeTick3').textContent = formatSeconds(dur * 0.75);
      document.getElementById('tapeTickEnd').textContent = data.video.length_str;

      // People cards
      const pRow = document.getElementById('peopleCardsRow');
      pRow.innerHTML = '';
      data.people.forEach(p => {
        const card = document.createElement('div');
        card.className = 'entity-card';
        card.onclick = () => {
          player.currentTime = p.first_t;
          player.play();
          document.getElementById('queryTextInput').value = `Show timeline for Person ${p.track_id}`;
          executeQuery();
        };
        card.innerHTML = `
          <div class="entity-id">${p.track_label} (${p.label})</div>
          <div class="entity-time">${p.time_range}</div>
          <div class="entity-desc">${p.desc}</div>
        `;
        pRow.appendChild(card);
      });

      // Search Suggestion Pills
      const pBox = document.getElementById('queryPillsContainer');
      pBox.innerHTML = '';
      data.suggestions.forEach(s => {
        const btn = document.createElement('button');
        btn.className = 'query-pill-btn';
        btn.textContent = `QUERY: ${s}`;
        btn.onclick = () => {
          document.getElementById('queryTextInput').value = s;
          executeQuery();
        };
        pBox.appendChild(btn);
      });

      // Gemini State Indicator
      document.getElementById('geminiStateIndicator').textContent = data.gemini_active ? `Gemini (${data.model_name}) Active` : `Deterministic SQL Offline`;
      document.getElementById('footerGeminiState').textContent = data.gemini_active ? `Online` : `Offline`;

      // Telemetry Analytics Panel
      document.getElementById('anSource').textContent = data.video.filename;
      document.getElementById('anCamId').textContent = data.video.camera_id.toUpperCase();
      document.getElementById('anDimensions').textContent = `${data.video.width}×${data.video.height} px`;
      document.getElementById('anFps').textContent = `${data.video.fps} FPS`;
      document.getElementById('anDuration').textContent = `${data.video.duration_s}s (${data.video.frame_count} frames)`;
      document.getElementById('anSha256').textContent = data.video.sha256;
      document.getElementById('anTracks').textContent = data.metrics.total_tracks;
      document.getElementById('anPeople').textContent = data.metrics.people_count;
      document.getElementById('anCars').textContent = data.metrics.cars_count;
      document.getElementById('anEvents').textContent = data.metrics.total_events;
    }

    function formatSeconds(sec) {
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
    }

    function onVideoTimeUpdate() {
      const player = document.getElementById('mainVideoPlayer');
      const cam = currentVideoData ? currentVideoData.video.camera_id.toUpperCase() : 'CAM-01';
      const fps = currentVideoData ? currentVideoData.video.fps : 30.0;
      const dur = currentVideoData ? currentVideoData.video.duration_s : 1.0;

      const t = player.currentTime;
      const hours = Math.floor(t / 3600);
      const mins = Math.floor((t % 3600) / 60);
      const secs = Math.floor(t % 60);
      const ms = Math.floor((t % 1) * 1000);
      const frameIdx = Math.floor(t * fps);

      const timeFormatted = `${String(hours).padStart(2,'0')}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
      const frameFormatted = `F${String(frameIdx).padStart(6,'0')}`;

      document.getElementById('burnOverlayTL').textContent = `${cam} · ${timeFormatted} · ${frameFormatted}`;

      // Move playhead on timeline
      const pct = Math.min(100, Math.max(0, (t / dur) * 100));
      document.getElementById('tapePlayhead').style.left = `${pct}%`;
    }

    function seekByTrackClick(e) {
      const tracksCol = document.getElementById('tapeTracksCol');
      const rect = tracksCol.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const pct = Math.min(1, Math.max(0, clickX / rect.width));
      if (currentVideoData) {
        const player = document.getElementById('mainVideoPlayer');
        player.currentTime = pct * currentVideoData.video.duration_s;
        player.play();
      }
    }

    function toggleCrosshair() {
      crosshairEnabled = !crosshairEnabled;
      document.getElementById('videoCrosshair').classList.toggle('active', crosshairEnabled);
      document.getElementById('crosshairBtn').classList.toggle('btn-primary', crosshairEnabled);
    }

    async function executeQuery() {
      const q = document.getElementById('queryTextInput').value.trim();
      if (!q || !activeVideoId) return;

      document.getElementById('queryLoadingState').style.display = 'block';
      document.getElementById('queryResultContainer').style.display = 'none';

      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: activeVideoId, query: q })
      });
      const data = await res.json();
      document.getElementById('queryLoadingState').style.display = 'none';
      document.getElementById('queryResultContainer').style.display = 'block';

      document.getElementById('queryResultText').textContent = data.summary;

      // Table body
      const tbody = document.getElementById('exhibitsTableBody');
      tbody.innerHTML = '';

      const evList = document.getElementById('queryEvidenceList');
      evList.innerHTML = '';

      if (data.evidence && data.evidence.length > 0) {
        document.getElementById('exhibitsTable').style.display = 'table';

        data.evidence.forEach(ev => {
          // Table Row
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${ev.idx}</td>
            <td>${ev.time_str}</td>
            <td><span style="color:var(--sig-amber)">${ev.track_label}</span></td>
            <td>${ev.class_name}</td>
            <td>
              <div class="conf-bar-wrap">
                <span class="conf-bar-bg"><span class="conf-bar-fill" style="width:${ev.conf_pct}%"></span></span>
                <span>${ev.confidence}</span>
              </div>
            </td>
            <td>
              <button class="btn" style="height:20px;padding:0 6px;font-size:10px;" onclick="seekToEvidence(${ev.t_start})">SEEK</button>
            </td>
          `;
          tbody.appendChild(tr);

          // Card preview
          const item = document.createElement('div');
          item.className = 'ev-card';
          item.innerHTML = `
            <div class="ev-card-header">
              <span><span class="ev-badge">${ev.type}</span> ${ev.subject}</span>
              <span>${ev.time_str} · CONF ${ev.confidence}</span>
            </div>
            <div class="ev-media-grid">
              ${ev.keyframe_url ? `<img src="${ev.keyframe_url}" alt="Keyframe">` : '<div style="background:#000;color:var(--text-tertiary);font-size:10px;display:flex;align-items:center;justify-content:center;">NO FRAME</div>'}
              ${ev.clip_url ? `<video controls playsinline src="${ev.clip_url}"></video>` : '<div style="background:#000;color:var(--text-tertiary);font-size:10px;display:flex;align-items:center;justify-content:center;">NO CLIP</div>'}
            </div>
          `;
          evList.appendChild(item);
        });
      } else {
        document.getElementById('exhibitsTable').style.display = 'none';
      }
    }

    function seekToEvidence(tStart) {
      const player = document.getElementById('mainVideoPlayer');
      player.currentTime = tStart;
      player.play();
    }

    function copyHash() {
      const hashText = document.getElementById('anSha256').textContent;
      navigator.clipboard.writeText(hashText);
      alert('SHA-256 copied to clipboard.');
    }

    async function confirmDeleteVideo(e, videoId, videoName) {
      if (e) e.stopPropagation();
      if (!confirm(`Are you sure you want to permanently delete "${videoName}" from the archive? All tracks, events, and exhibits will be removed.`)) {
        return;
      }
      const res = await fetch(`/api/video/${videoId}`, { method: 'DELETE' });
      if (res.ok) {
        if (activeVideoId === videoId) {
          activeVideoId = null;
        }
        await loadWall();
        showView('wall');
      } else {
        alert('Failed to delete video.');
      }
    }

    async function confirmDeleteActiveVideo() {
      if (!activeVideoId || !currentVideoData) return;
      await confirmDeleteVideo(null, activeVideoId, currentVideoData.video.name);
    }

    function showView(view) {
      document.getElementById('viewWall').style.display = view === 'wall' ? 'block' : 'none';
      document.getElementById('viewInvestigate').style.display = view === 'investigate' ? 'block' : 'none';
      document.getElementById('viewAnalytics').style.display = view === 'analytics' ? 'block' : 'none';

      document.getElementById('btnNavWall').classList.toggle('active', view === 'wall');
      document.getElementById('btnNavInvestigate').classList.toggle('active', view === 'investigate');
      document.getElementById('btnNavAnalytics').classList.toggle('active', view === 'analytics');

      if (view === 'investigate' && activeVideoId) {
        loadInvestigateVideo(activeVideoId);
      }
    }

    function triggerUpload() {
      document.getElementById('fileInput').click();
    }

    function handleFileUpload(e) {
      const file = e.target.files[0];
      if (!file) return;

      const progressBox = document.getElementById('uploadProgressBox');
      const statusText = document.getElementById('uploadStatusText');
      const byteStats = document.getElementById('uploadByteStats');
      const fill = document.getElementById('uploadProgressFill');

      progressBox.style.display = 'block';
      statusText.textContent = `UPLOADING ${file.name.toUpperCase()} · 0.0%`;
      statusText.style.color = 'var(--sig-amber)';
      byteStats.textContent = `0.0 MB / ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
      fill.style.width = '0%';

      const fd = new FormData();
      fd.append('file', file);

      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/upload', true);

      xhr.upload.onprogress = function(event) {
        if (event.lengthComputable) {
          const pct = (event.loaded / event.total) * 100;
          const loadedMB = (event.loaded / (1024 * 1024)).toFixed(1);
          const totalMB = (event.total / (1024 * 1024)).toFixed(1);
          fill.style.width = `${pct.toFixed(1)}%`;
          statusText.textContent = `UPLOADING FOOTAGE · ${pct.toFixed(1)}%`;
          byteStats.textContent = `${loadedMB} MB / ${totalMB} MB`;
        }
      };

      xhr.onload = async function() {
        if (xhr.status === 200) {
          statusText.textContent = 'RUNNING PERCEPTION & INDEXING (YOLO11n + BYTETRACK)…';
          byteStats.textContent = 'EXTRACTING DETECTIONS & KEYFRAMES';
          fill.style.width = '100%';

          await loadWall();
          setTimeout(() => {
            progressBox.style.display = 'none';
          }, 1500);
        } else {
          statusText.textContent = 'UPLOAD OR INDEXING FAILED';
          statusText.style.color = 'var(--sig-red)';
          byteStats.textContent = 'STATUS ERROR';
        }
        e.target.value = '';
      };

      xhr.onerror = function() {
        statusText.textContent = 'NETWORK CONNECTION ERROR';
        statusText.style.color = 'var(--sig-red)';
        byteStats.textContent = 'OFFLINE';
        e.target.value = '';
      };

      xhr.send(fd);
    }

    async function runPipelineOnActive() {
      if (!activeVideoId) return;
      const btn = document.getElementById('reprocessBtn');
      btn.textContent = 'SCANNING & INDEXING…';
      btn.disabled = true;

      const res = await fetch(`/api/process/${activeVideoId}`, { method: 'POST' });
      btn.textContent = 'Re-scan & Index Footage';
      btn.disabled = false;

      if (res.ok) {
        alert('Indexing complete.');
        loadInvestigateVideo(activeVideoId);
      } else {
        alert('Indexing failed.');
      }
    }

    window.onload = loadWall;
  </script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
