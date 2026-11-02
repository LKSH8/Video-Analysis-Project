-- SQLite Database Schema for Semantic CCTV Event Retrieval

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    camera_id TEXT NOT NULL DEFAULT 'cam1',
    fps REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    duration_s REAL NOT NULL,
    start_wallclock TEXT,
    processed_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS tracks (
    video_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    class TEXT NOT NULL,
    first_t REAL NOT NULL,
    last_t REAL NOT NULL,
    n_frames INTEGER NOT NULL,
    best_conf REAL NOT NULL,
    best_frame_t REAL NOT NULL,
    color TEXT,
    PRIMARY KEY (video_id, track_id),
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    frame INTEGER NOT NULL,
    t REAL NOT NULL,
    track_id INTEGER NOT NULL,
    class TEXT NOT NULL,
    conf REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_detections_vtrack ON detections(video_id, track_id);
CREATE INDEX IF NOT EXISTS idx_detections_vt ON detections(video_id, t);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    camera_id TEXT NOT NULL DEFAULT 'cam1',
    type TEXT NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    track_id INTEGER NOT NULL,
    object_track_id INTEGER,
    class TEXT NOT NULL,
    object_class TEXT,
    confidence REAL NOT NULL,
    details_json TEXT,
    keyframe_path TEXT,
    clip_path TEXT,
    embedding BLOB,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(video_id, type);
CREATE INDEX IF NOT EXISTS idx_events_tstart ON events(video_id, t_start);
CREATE INDEX IF NOT EXISTS idx_events_class ON events(video_id, class);

CREATE TABLE IF NOT EXISTS frame_stats (
    video_id INTEGER NOT NULL,
    t REAL NOT NULL,
    n_people INTEGER NOT NULL,
    n_objects INTEGER NOT NULL,
    PRIMARY KEY (video_id, t),
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);
