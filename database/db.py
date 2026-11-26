import sqlite3
import os
from pathlib import Path
from config import DB_PATH

def get_db_connection():
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes all required surveillance schema tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        storage_path TEXT NOT NULL,
        duration REAL,
        fps REAL,
        total_frames INTEGER,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT CHECK(status IN ('uploaded', 'processing', 'completed', 'failed'))
    );

    CREATE TABLE IF NOT EXISTS frames (
        frame_id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        camera_id TEXT DEFAULT 'cam_01',
        frame_number INTEGER NOT NULL,
        timestamp_sec REAL NOT NULL,
        storage_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS detections (
        detection_id TEXT PRIMARY KEY,
        frame_id TEXT NOT NULL,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox_x1 REAL NOT NULL,
        bbox_y1 REAL NOT NULL,
        bbox_x2 REAL NOT NULL,
        bbox_y2 REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        embedding_id TEXT PRIMARY KEY,
        frame_id TEXT NOT NULL UNIQUE,
        vector_faiss_id INTEGER NOT NULL UNIQUE,
        is_encrypted INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS alerts (
        alert_id TEXT PRIMARY KEY,
        rule_id TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        frame_id TEXT,
        details TEXT,
        is_demo INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
    );

    CREATE TABLE IF NOT EXISTS person_tracks (
        track_id TEXT PRIMARY KEY,
        global_person_id TEXT NOT NULL,
        detection_id TEXT NOT NULL,
        frame_id TEXT NOT NULL,
        video_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        timestamp_sec REAL NOT NULL,
        crop_path TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE,
        FOREIGN KEY(detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_frames_video ON frames(video_id);
    CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_id);
    CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);
    CREATE INDEX IF NOT EXISTS idx_embeddings_frame ON embeddings(frame_id);
    CREATE INDEX IF NOT EXISTS idx_tracks_person ON person_tracks(global_person_id);
    CREATE INDEX IF NOT EXISTS idx_tracks_camera ON person_tracks(camera_id);
    CREATE INDEX IF NOT EXISTS idx_tracks_video ON person_tracks(video_id);
    CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id);
    CREATE INDEX IF NOT EXISTS idx_alerts_rule ON alerts(rule_id);
    """)

    # Column migrations for alerts table if needed
    try:
        cursor.execute("ALTER TABLE alerts ADD COLUMN severity TEXT DEFAULT 'HIGH'")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE alerts ADD COLUMN status TEXT DEFAULT 'active'")
    except Exception:
        pass

    conn.commit()
    conn.close()
    return True

def insert_video(video_id, filename, storage_path, duration=0.0, fps=0.0, total_frames=0, status="uploaded"):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO videos (video_id, filename, storage_path, duration, fps, total_frames, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (video_id, filename, storage_path, duration, fps, total_frames, status))
    conn.commit()
    conn.close()

def update_video_status(video_id, status, duration=None, fps=None, total_frames=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if duration is not None and fps is not None and total_frames is not None:
        cursor.execute("""
            UPDATE videos SET status = ?, duration = ?, fps = ?, total_frames = ? WHERE video_id = ?
        """, (status, duration, fps, total_frames, video_id))
    else:
        cursor.execute("UPDATE videos SET status = ? WHERE video_id = ?", (status, video_id))
    conn.commit()
    conn.close()

def get_all_videos():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM videos ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def insert_frames_batch(frames_data):
    """
    frames_data: list of tuples:
    [(frame_id, video_id, camera_id, frame_number, timestamp_sec, storage_path), ...]
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO frames (frame_id, video_id, camera_id, frame_number, timestamp_sec, storage_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, frames_data)
    conn.commit()
    conn.close()

def insert_detections_batch(detections_data):
    """
    detections_data: list of tuples:
    [(detection_id, frame_id, class_name, confidence, x1, y1, x2, y2), ...]
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO detections (detection_id, frame_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, detections_data)
    conn.commit()
    conn.close()

def get_video_by_id(video_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_frames_by_video(video_id):
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM frames WHERE video_id = ? ORDER BY timestamp_sec ASC", (video_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_detections_by_frame(frame_id):
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM detections WHERE frame_id = ?", (frame_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def insert_embeddings_batch(embeddings_data):
    """
    embeddings_data: list of tuples:
    [(embedding_id, frame_id, vector_faiss_id, is_encrypted), ...]
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO embeddings (embedding_id, frame_id, vector_faiss_id, is_encrypted)
        VALUES (?, ?, ?, ?)
    """, embeddings_data)
    conn.commit()
    conn.close()

def get_embeddings_encryption_status():
    """Week 9: Returns count of total and encrypted embedding records."""
    conn = get_db_connection()
    total = conn.execute("SELECT COUNT(*) as c FROM embeddings").fetchone()['c']
    encrypted = conn.execute("SELECT COUNT(*) as c FROM embeddings WHERE is_encrypted = 1").fetchone()['c']
    conn.close()
    return {"total_embeddings": total, "encrypted_embeddings": encrypted}

def update_all_embeddings_encrypted_flag(is_encrypted=1):
    """Week 9: Sets encryption status for all stored embedding records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE embeddings SET is_encrypted = ?", (is_encrypted,))
    conn.commit()
    conn.close()

def get_unembedded_frames():
    """Retrieves frames that have not yet been indexed in the vector store."""
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT f.frame_id, f.video_id, f.timestamp_sec, f.storage_path
        FROM frames f
        LEFT JOIN embeddings e ON f.frame_id = e.frame_id
        WHERE e.embedding_id IS NULL
        ORDER BY f.created_at ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_frames():
    """Retrieves all frames with video metadata."""
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT f.frame_id, f.video_id, f.timestamp_sec, f.storage_path, v.filename
        FROM frames f
        JOIN videos v ON f.video_id = v.video_id
        ORDER BY f.timestamp_sec ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_frame_details_for_search(frame_id):
    """P2.7: SQLite lookup for frame, video, and detections."""
    conn = get_db_connection()
    row = conn.execute("""
        SELECT f.frame_id, f.storage_path, f.timestamp_sec, f.camera_id,
               v.video_id, v.filename, v.storage_path AS video_path
        FROM frames f
        JOIN videos v ON f.video_id = v.video_id
        WHERE f.frame_id = ?
    """, (frame_id,)).fetchone()
    
    if not row:
        conn.close()
        return None

    frame_info = dict(row)
    det_rows = conn.execute("""
        SELECT class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM detections
        WHERE frame_id = ?
        ORDER BY confidence DESC
    """, (frame_id,)).fetchall()
    
    frame_info["detections"] = [dict(d) for d in det_rows]
    conn.close()
    return frame_info

def delete_video_cascade(video_id: str):
    """
    Safely cascades deletion for an ingested video and its related frames,
    detections, embeddings, alerts, and person tracks.
    Returns (video_info, deleted_frame_ids, stats).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    video_row = cursor.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    if not video_row:
        conn.close()
        return None, [], {}

    video_info = dict(video_row)

    frame_rows = cursor.execute("SELECT frame_id, storage_path FROM frames WHERE video_id = ?", (video_id,)).fetchall()
    frame_ids = [r["frame_id"] for r in frame_rows]
    frame_paths = [r["storage_path"] for r in frame_rows]

    det_count = 0
    emb_count = 0
    track_count = 0

    if frame_ids:
        placeholders = ",".join("?" for _ in frame_ids)
        det_count = cursor.execute(f"SELECT COUNT(*) FROM detections WHERE frame_id IN ({placeholders})", frame_ids).fetchone()[0]
        emb_count = cursor.execute(f"SELECT COUNT(*) FROM embeddings WHERE frame_id IN ({placeholders})", frame_ids).fetchone()[0]
        
        try:
            track_count = cursor.execute(f"SELECT COUNT(*) FROM person_tracks WHERE frame_id IN ({placeholders})", frame_ids).fetchone()[0]
            cursor.execute(f"DELETE FROM person_tracks WHERE frame_id IN ({placeholders})", frame_ids)
        except Exception:
            track_count = 0

        cursor.execute(f"DELETE FROM alerts WHERE frame_id IN ({placeholders})", frame_ids)
        cursor.execute(f"DELETE FROM embeddings WHERE frame_id IN ({placeholders})", frame_ids)
        cursor.execute(f"DELETE FROM detections WHERE frame_id IN ({placeholders})", frame_ids)

    cursor.execute("DELETE FROM frames WHERE video_id = ?", (video_id,))
    cursor.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))

    conn.commit()
    conn.close()

    stats = {
        "deleted_frames_count": len(frame_ids),
        "deleted_detections_count": det_count,
        "deleted_embeddings_count": emb_count,
        "deleted_tracks_count": track_count,
        "frame_paths": frame_paths
    }
    return video_info, frame_ids, stats

# ==========================================
# Person Re-Identification Database Methods (Week 7)
# ==========================================

def insert_person_track(track_id, global_person_id, detection_id, frame_id, video_id, camera_id, timestamp_sec, crop_path, confidence):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO person_tracks 
        (track_id, global_person_id, detection_id, frame_id, video_id, camera_id, timestamp_sec, crop_path, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (track_id, global_person_id, detection_id, frame_id, video_id, camera_id, timestamp_sec, crop_path, confidence))
    conn.commit()
    conn.close()

def get_all_person_tracks(camera_id=None):
    conn = get_db_connection()
    if camera_id:
        rows = conn.execute("""
            SELECT pt.*, v.filename
            FROM person_tracks pt
            JOIN videos v ON pt.video_id = v.video_id
            WHERE pt.camera_id = ?
            ORDER BY pt.timestamp_sec ASC
        """, (camera_id,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT pt.*, v.filename
            FROM person_tracks pt
            JOIN videos v ON pt.video_id = v.video_id
            ORDER BY pt.global_person_id ASC, pt.timestamp_sec ASC
        """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_person_track_by_id(track_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM person_tracks WHERE track_id = ?", (track_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_tracks_by_global_id(global_person_id):
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT pt.*, v.filename
        FROM person_tracks pt
        JOIN videos v ON pt.video_id = v.video_id
        WHERE pt.global_person_id = ?
        ORDER BY pt.timestamp_sec ASC
    """, (global_person_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_unique_global_persons():
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT global_person_id, 
               COUNT(*) as appearances, 
               COUNT(DISTINCT camera_id) as camera_count,
               MIN(timestamp_sec) as first_seen,
               MAX(timestamp_sec) as last_seen,
               crop_path as representative_crop,
               camera_id as first_camera
        FROM person_tracks
        GROUP BY global_person_id
        ORDER BY appearances DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ==========================================
# Alert Engine Database Methods (Week 8)
# ==========================================

def insert_alert(alert_id, rule_id, alert_type, camera_id, frame_id, details, severity="HIGH", status="active", is_demo=0):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts (alert_id, rule_id, alert_type, camera_id, frame_id, details, severity, status, is_demo)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (alert_id, rule_id, alert_type, camera_id, frame_id, details, severity, status, 1 if is_demo else 0))
    conn.commit()
    conn.close()

def get_all_alerts(limit=100, status=None):
    conn = get_db_connection()
    if status:
        rows = conn.execute("""
            SELECT a.*, f.video_id, f.timestamp_sec, f.storage_path, v.filename as video_name
            FROM alerts a
            LEFT JOIN frames f ON a.frame_id = f.frame_id
            LEFT JOIN videos v ON f.video_id = v.video_id
            WHERE a.status = ?
            ORDER BY a.created_at DESC
            LIMIT ?
        """, (status, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT a.*, f.video_id, f.timestamp_sec, f.storage_path, v.filename as video_name
            FROM alerts a
            LEFT JOIN frames f ON a.frame_id = f.frame_id
            LEFT JOIN videos v ON f.video_id = v.video_id
            ORDER BY a.created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_alert_by_id(alert_id):
    conn = get_db_connection()
    row = conn.execute("""
        SELECT a.*, f.video_id, f.timestamp_sec, f.storage_path, v.filename as video_name
        FROM alerts a
        LEFT JOIN frames f ON a.frame_id = f.frame_id
        LEFT JOIN videos v ON f.video_id = v.video_id
        WHERE a.alert_id = ?
    """, (alert_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def update_alert_status(alert_id, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE alerts SET status = ? WHERE alert_id = ?", (status, alert_id))
    conn.commit()
    conn.close()

def get_alerts_summary():
    conn = get_db_connection()
    total = conn.execute("SELECT COUNT(*) as c FROM alerts").fetchone()['c']
    active = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE status = 'active'").fetchone()['c']
    critical = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE severity = 'CRITICAL' AND status = 'active'").fetchone()['c']
    high = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE severity = 'HIGH' AND status = 'active'").fetchone()['c']
    demo_count = conn.execute("SELECT COUNT(*) as c FROM alerts WHERE is_demo = 1").fetchone()['c']
    by_rule = conn.execute("""
        SELECT rule_id, alert_type, COUNT(*) as count 
        FROM alerts 
        GROUP BY rule_id 
        ORDER BY count DESC
    """).fetchall()
    conn.close()
    return {
        "total_alerts": total,
        "active_alerts": active,
        "critical_active": critical,
        "high_active": high,
        "demo_alerts": demo_count,
        "rule_breakdown": [dict(r) for r in by_rule]
    }

def clear_all_alerts(demo_only=False):
    conn = get_db_connection()
    cursor = conn.cursor()
    if demo_only:
        cursor.execute("DELETE FROM alerts WHERE is_demo = 1")
    else:
        cursor.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
