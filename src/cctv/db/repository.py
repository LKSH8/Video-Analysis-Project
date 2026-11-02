"""Repository pattern abstraction and SQLite implementation.

All database queries and SQL statements live strictly in this file.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional


@dataclass
class VideoRecord:
    id: int
    path: str
    camera_id: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float
    start_wallclock: Optional[str]
    processed_at: Optional[str]
    status: str


@dataclass
class TrackRecord:
    video_id: int
    track_id: int
    class_name: str
    first_t: float
    last_t: float
    n_frames: int
    best_conf: float
    best_frame_t: float
    color: Optional[str] = None


@dataclass
class DetectionRecord:
    video_id: int
    frame: int
    t: float
    track_id: int
    class_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    id: Optional[int] = None


@dataclass
class EventRecord:
    video_id: int
    camera_id: str
    type: str
    t_start: float
    t_end: float
    track_id: int
    class_name: str
    confidence: float
    object_track_id: Optional[int] = None
    object_class: Optional[str] = None
    details_json: Optional[str] = None
    keyframe_path: Optional[str] = None
    clip_path: Optional[str] = None
    embedding: Optional[bytes] = None
    id: Optional[int] = None


@dataclass
class FrameStatRecord:
    video_id: int
    t: float
    n_people: int
    n_objects: int


class Repository(ABC):
    """Abstract interface defining all data storage and retrieval capabilities."""

    @abstractmethod
    def init_schema(self) -> None:
        pass

    @abstractmethod
    def get_or_create_video(
        self,
        path: str,
        camera_id: str,
        fps: float,
        width: int,
        height: int,
        frame_count: int,
        duration_s: float,
        start_wallclock: Optional[str] = None,
    ) -> VideoRecord:
        pass

    @abstractmethod
    def get_video(self, video_id: int) -> Optional[VideoRecord]:
        pass

    @abstractmethod
    def get_video_by_path(self, path: str) -> Optional[VideoRecord]:
        pass

    @abstractmethod
    def list_videos(self) -> List[VideoRecord]:
        pass

    @abstractmethod
    def update_video_status(
        self, video_id: int, status: str, processed_at: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def clear_video_data(self, video_id: int) -> None:
        pass

    @abstractmethod
    def insert_tracks(self, tracks: List[TrackRecord]) -> None:
        pass

    @abstractmethod
    def insert_detections(self, detections: List[DetectionRecord]) -> None:
        pass

    @abstractmethod
    def insert_events(self, events: List[EventRecord]) -> None:
        pass

    @abstractmethod
    def insert_frame_stats(self, stats: List[FrameStatRecord]) -> None:
        pass

    @abstractmethod
    def get_tracks(
        self,
        video_id: int,
        track_id: Optional[int] = None,
        class_name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> List[TrackRecord]:
        pass

    @abstractmethod
    def get_detections(
        self,
        video_id: int,
        track_id: Optional[int] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> List[DetectionRecord]:
        pass

    @abstractmethod
    def query_events(
        self,
        video_id: int,
        event_types: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        track_id: Optional[int] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        limit: int = 20,
        order_desc: bool = False,
    ) -> List[EventRecord]:
        pass

    @abstractmethod
    def get_anchor_event(
        self,
        video_id: int,
        event_type: str,
        class_name: Optional[str] = None,
        order: str = "first",
    ) -> Optional[EventRecord]:
        pass

    @abstractmethod
    def count_events(
        self,
        video_id: int,
        event_types: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        track_id: Optional[int] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> int:
        pass

    @abstractmethod
    def count_tracks(
        self, video_id: int, class_name: Optional[str] = None
    ) -> int:
        pass

    @abstractmethod
    def get_frame_stats(self, video_id: int) -> List[FrameStatRecord]:
        pass

    @abstractmethod
    def get_event_counts_by_type(self, video_id: int) -> Dict[str, int]:
        pass

    @abstractmethod
    def get_distinct_classes_in_video(self, video_id: int) -> List[str]:
        pass

    @abstractmethod
    def get_distinct_event_types_in_video(self, video_id: int) -> List[str]:
        pass


class SQLiteRepository(Repository):
    """SQLite implementation of the Repository interface."""

    def __init__(self, db_path: str = "data/cctv.db", schema_path: Optional[str] = None):
        self.db_path = db_path
        if schema_path is None:
            schema_path = str(Path(__file__).parent / "schema.sql")
        self.schema_path = schema_path
        
        # Ensure parent directory exists for file-based DB
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
        self._conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            # Keep persistent connection for in-memory database
            self._conn = sqlite3.connect(":memory:")
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self.init_schema()
        else:
            self.init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_schema(self) -> None:
        """Initialize database tables and indexes from schema.sql."""
        schema_file = Path(self.schema_path)
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found at {self.schema_path}")
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn = self._get_connection()
        try:
            conn.executescript(schema_sql)
            # Safe migration: ensure 'color' column exists in tracks table
            try:
                conn.execute("ALTER TABLE tracks ADD COLUMN color TEXT;")
            except Exception:
                pass
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def get_or_create_video(
        self,
        path: str,
        camera_id: str,
        fps: float,
        width: int,
        height: int,
        frame_count: int,
        duration_s: float,
        start_wallclock: Optional[str] = None,
    ) -> VideoRecord:
        normalized_path = str(Path(path).as_posix())
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE path = ?", (normalized_path,))
            row = cursor.fetchone()
            if row:
                # Update existing row with latest metadata
                cursor.execute(
                    """
                    UPDATE videos
                    SET camera_id = ?, fps = ?, width = ?, height = ?, frame_count = ?,
                        duration_s = ?, start_wallclock = coalesce(?, start_wallclock)
                    WHERE id = ?
                    """,
                    (camera_id, fps, width, height, frame_count, duration_s, start_wallclock, row["id"]),
                )
                conn.commit()
                return self.get_video(row["id"])  # type: ignore

            cursor.execute(
                """
                INSERT INTO videos (path, camera_id, fps, width, height, frame_count, duration_s, start_wallclock, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (normalized_path, camera_id, fps, width, height, frame_count, duration_s, start_wallclock),
            )
            video_id = cursor.lastrowid
            conn.commit()
            return self.get_video(video_id)  # type: ignore
        finally:
            if self._conn is None:
                conn.close()

    def get_video(self, video_id: int) -> Optional[VideoRecord]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return VideoRecord(
                id=row["id"],
                path=row["path"],
                camera_id=row["camera_id"],
                fps=row["fps"],
                width=row["width"],
                height=row["height"],
                frame_count=row["frame_count"],
                duration_s=row["duration_s"],
                start_wallclock=row["start_wallclock"],
                processed_at=row["processed_at"],
                status=row["status"],
            )
        finally:
            if self._conn is None:
                conn.close()

    def get_video_by_path(self, path: str) -> Optional[VideoRecord]:
        normalized_path = str(Path(path).as_posix())
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos WHERE path = ?", (normalized_path,))
            row = cursor.fetchone()
            if not row:
                return None
            return VideoRecord(
                id=row["id"],
                path=row["path"],
                camera_id=row["camera_id"],
                fps=row["fps"],
                width=row["width"],
                height=row["height"],
                frame_count=row["frame_count"],
                duration_s=row["duration_s"],
                start_wallclock=row["start_wallclock"],
                processed_at=row["processed_at"],
                status=row["status"],
            )
        finally:
            if self._conn is None:
                conn.close()

    def list_videos(self) -> List[VideoRecord]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM videos ORDER BY id DESC")
            rows = cursor.fetchall()
            return [
                VideoRecord(
                    id=row["id"],
                    path=row["path"],
                    camera_id=row["camera_id"],
                    fps=row["fps"],
                    width=row["width"],
                    height=row["height"],
                    frame_count=row["frame_count"],
                    duration_s=row["duration_s"],
                    start_wallclock=row["start_wallclock"],
                    processed_at=row["processed_at"],
                    status=row["status"],
                )
                for row in rows
            ]
        finally:
            if self._conn is None:
                conn.close()

    def update_video_status(
        self, video_id: int, status: str, processed_at: Optional[str] = None
    ) -> None:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if processed_at:
                cursor.execute(
                    "UPDATE videos SET status = ?, processed_at = ? WHERE id = ?",
                    (status, processed_at, video_id),
                )
            else:
                cursor.execute(
                    "UPDATE videos SET status = ? WHERE id = ?",
                    (status, video_id),
                )
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def clear_video_data(self, video_id: int) -> None:
        """Idempotently clear all extracted data for a video."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections WHERE video_id = ?", (video_id,))
            cursor.execute("DELETE FROM tracks WHERE video_id = ?", (video_id,))
            cursor.execute("DELETE FROM events WHERE video_id = ?", (video_id,))
            cursor.execute("DELETE FROM frame_stats WHERE video_id = ?", (video_id,))
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def delete_video(self, video_id: int) -> None:
        """Completely delete video record and all associated tracks, events, and detections."""
        self.clear_video_data(video_id)
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM videos WHERE id = ?", (video_id,))
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def insert_tracks(self, tracks: List[TrackRecord]) -> None:
        if not tracks:
            return
        conn = self._get_connection()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO tracks
                (video_id, track_id, class, first_t, last_t, n_frames, best_conf, best_frame_t, color)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        t.video_id,
                        t.track_id,
                        t.class_name,
                        t.first_t,
                        t.last_t,
                        t.n_frames,
                        t.best_conf,
                        t.best_frame_t,
                        t.color,
                    )
                    for t in tracks
                ],
            )
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def insert_detections(self, detections: List[DetectionRecord]) -> None:
        if not detections:
            return
        conn = self._get_connection()
        try:
            conn.executemany(
                """
                INSERT INTO detections
                (video_id, frame, t, track_id, class, conf, x1, y1, x2, y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        d.video_id,
                        d.frame,
                        d.t,
                        d.track_id,
                        d.class_name,
                        d.conf,
                        d.x1,
                        d.y1,
                        d.x2,
                        d.y2,
                    )
                    for d in detections
                ],
            )
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def insert_events(self, events: List[EventRecord]) -> None:
        if not events:
            return
        conn = self._get_connection()
        try:
            conn.executemany(
                """
                INSERT INTO events
                (video_id, camera_id, type, t_start, t_end, track_id, object_track_id,
                 class, object_class, confidence, details_json, keyframe_path, clip_path, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.video_id,
                        e.camera_id,
                        e.type,
                        e.t_start,
                        e.t_end,
                        e.track_id,
                        e.object_track_id,
                        e.class_name,
                        e.object_class,
                        e.confidence,
                        e.details_json,
                        e.keyframe_path,
                        e.clip_path,
                        e.embedding,
                    )
                    for e in events
                ],
            )
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def insert_frame_stats(self, stats: List[FrameStatRecord]) -> None:
        if not stats:
            return
        conn = self._get_connection()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO frame_stats
                (video_id, t, n_people, n_objects)
                VALUES (?, ?, ?, ?)
                """,
                [(s.video_id, s.t, s.n_people, s.n_objects) for s in stats],
            )
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def get_tracks(
        self,
        video_id: int,
        track_id: Optional[int] = None,
        class_name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> List[TrackRecord]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM tracks WHERE video_id = ?"
            params: List[Any] = [video_id]
            if track_id is not None:
                query += " AND track_id = ?"
                params.append(track_id)
            if class_name is not None:
                query += " AND class = ?"
                params.append(class_name)
            if color is not None:
                query += " AND color = ?"
                params.append(color)
            query += " ORDER BY first_t ASC"

            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                TrackRecord(
                    video_id=r["video_id"],
                    track_id=r["track_id"],
                    class_name=r["class"],
                    first_t=r["first_t"],
                    last_t=r["last_t"],
                    n_frames=r["n_frames"],
                    best_conf=r["best_conf"],
                    best_frame_t=r["best_frame_t"],
                    color=r["color"] if "color" in r.keys() else None,
                )
                for r in rows
            ]
        finally:
            if self._conn is None:
                conn.close()

    def get_detections(
        self,
        video_id: int,
        track_id: Optional[int] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> List[DetectionRecord]:
        conn = self._get_connection()
        try:
            query = "SELECT * FROM detections WHERE video_id = ?"
            params: List[Any] = [video_id]
            if track_id is not None:
                query += " AND track_id = ?"
                params.append(track_id)
            if t_start is not None:
                query += " AND t >= ?"
                params.append(t_start)
            if t_end is not None:
                query += " AND t <= ?"
                params.append(t_end)
            query += " ORDER BY t ASC"

            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                DetectionRecord(
                    id=r["id"],
                    video_id=r["video_id"],
                    frame=r["frame"],
                    t=r["t"],
                    track_id=r["track_id"],
                    class_name=r["class"],
                    conf=r["conf"],
                    x1=r["x1"],
                    y1=r["y1"],
                    x2=r["x2"],
                    y2=r["y2"],
                )
                for r in rows
            ]
        finally:
            if self._conn is None:
                conn.close()

    def query_events(
        self,
        video_id: int,
        event_types: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        track_id: Optional[int] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        limit: int = 20,
        order_desc: bool = False,
    ) -> List[EventRecord]:
        conn = self._get_connection()
        try:
            clauses = ["video_id = ?"]
            params: List[Any] = [video_id]

            if event_types:
                placeholders = ",".join("?" for _ in event_types)
                clauses.append(f"type IN ({placeholders})")
                params.extend(event_types)

            if classes:
                # Match either subject class or target object class
                class_clauses = []
                for c in classes:
                    class_clauses.append("(class = ? OR object_class = ?)")
                    params.extend([c, c])
                clauses.append(f"({' OR '.join(class_clauses)})")

            if track_id is not None:
                clauses.append("(track_id = ? OR object_track_id = ?)")
                params.extend([track_id, track_id])

            if t_start is not None:
                clauses.append("t_end >= ?")
                params.append(t_start)

            if t_end is not None:
                clauses.append("t_start <= ?")
                params.append(t_end)

            order = "DESC" if order_desc else "ASC"
            query = f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY t_start {order} LIMIT ?"
            params.append(limit)

            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                EventRecord(
                    id=r["id"],
                    video_id=r["video_id"],
                    camera_id=r["camera_id"],
                    type=r["type"],
                    t_start=r["t_start"],
                    t_end=r["t_end"],
                    track_id=r["track_id"],
                    object_track_id=r["object_track_id"],
                    class_name=r["class"],
                    object_class=r["object_class"],
                    confidence=r["confidence"],
                    details_json=r["details_json"],
                    keyframe_path=r["keyframe_path"],
                    clip_path=r["clip_path"],
                    embedding=r["embedding"],
                )
                for r in rows
            ]
        finally:
            if self._conn is None:
                conn.close()

    def get_anchor_event(
        self,
        video_id: int,
        event_type: str,
        class_name: Optional[str] = None,
        order: str = "first",
    ) -> Optional[EventRecord]:
        conn = self._get_connection()
        try:
            clauses = ["video_id = ?", "type = ?"]
            params: List[Any] = [video_id, event_type]
            if class_name:
                clauses.append("(class = ? OR object_class = ?)")
                params.extend([class_name, class_name])

            sort_order = "DESC" if order == "last" else "ASC"
            query = f"SELECT * FROM events WHERE {' AND '.join(clauses)} ORDER BY t_start {sort_order} LIMIT 1"

            cursor = conn.cursor()
            cursor.execute(query, params)
            r = cursor.fetchone()
            if not r:
                return None
            return EventRecord(
                id=r["id"],
                video_id=r["video_id"],
                camera_id=r["camera_id"],
                type=r["type"],
                t_start=r["t_start"],
                t_end=r["t_end"],
                track_id=r["track_id"],
                object_track_id=r["object_track_id"],
                class_name=r["class"],
                object_class=r["object_class"],
                confidence=r["confidence"],
                details_json=r["details_json"],
                keyframe_path=r["keyframe_path"],
                clip_path=r["clip_path"],
                embedding=r["embedding"],
            )
        finally:
            if self._conn is None:
                conn.close()

    def count_events(
        self,
        video_id: int,
        event_types: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        track_id: Optional[int] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> int:
        conn = self._get_connection()
        try:
            clauses = ["video_id = ?"]
            params: List[Any] = [video_id]

            if event_types:
                placeholders = ",".join("?" for _ in event_types)
                clauses.append(f"type IN ({placeholders})")
                params.extend(event_types)

            if classes:
                class_clauses = []
                for c in classes:
                    class_clauses.append("(class = ? OR object_class = ?)")
                    params.extend([c, c])
                clauses.append(f"({' OR '.join(class_clauses)})")

            if track_id is not None:
                clauses.append("(track_id = ? OR object_track_id = ?)")
                params.extend([track_id, track_id])

            if t_start is not None:
                clauses.append("t_end >= ?")
                params.append(t_start)

            if t_end is not None:
                clauses.append("t_start <= ?")
                params.append(t_end)

            query = f"SELECT COUNT(*) as cnt FROM events WHERE {' AND '.join(clauses)}"
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row["cnt"] if row else 0
        finally:
            if self._conn is None:
                conn.close()

    def count_tracks(
        self, video_id: int, class_name: Optional[str] = None
    ) -> int:
        conn = self._get_connection()
        try:
            query = "SELECT COUNT(*) as cnt FROM tracks WHERE video_id = ?"
            params: List[Any] = [video_id]
            if class_name:
                query += " AND class = ?"
                params.append(class_name)
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row["cnt"] if row else 0
        finally:
            if self._conn is None:
                conn.close()

    def get_frame_stats(self, video_id: int) -> List[FrameStatRecord]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM frame_stats WHERE video_id = ? ORDER BY t ASC",
                (video_id,),
            )
            rows = cursor.fetchall()
            return [
                FrameStatRecord(
                    video_id=r["video_id"],
                    t=r["t"],
                    n_people=r["n_people"],
                    n_objects=r["n_objects"],
                )
                for r in rows
            ]
        finally:
            if self._conn is None:
                conn.close()

    def get_event_counts_by_type(self, video_id: int) -> Dict[str, int]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT type, COUNT(*) as count FROM events WHERE video_id = ? GROUP BY type",
                (video_id,),
            )
            return {row["type"]: row["count"] for row in cursor.fetchall()}
        finally:
            if self._conn is None:
                conn.close()

    def get_distinct_classes_in_video(self, video_id: int) -> List[str]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT class FROM (
                    SELECT class FROM tracks WHERE video_id = ?
                    UNION
                    SELECT class FROM events WHERE video_id = ?
                    UNION
                    SELECT object_class FROM events WHERE video_id = ? AND object_class IS NOT NULL
                )
                """,
                (video_id, video_id, video_id),
            )
            return [r[0] for r in cursor.fetchall() if r[0]]
        finally:
            if self._conn is None:
                conn.close()

    def get_distinct_event_types_in_video(self, video_id: int) -> List[str]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT type FROM events WHERE video_id = ?",
                (video_id,),
            )
            return [r["type"] for r in cursor.fetchall()]
        finally:
            if self._conn is None:
                conn.close()
