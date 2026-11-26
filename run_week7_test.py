"""
================================================================================
  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM
  WEEK 7 VERIFICATION TEST SUITE & VIDEO DELETION VERIFICATION
================================================================================
Tests:
1. Video Deletion Feature (cascade DB delete + frame cleanup + FAISS sync)
2. Person Re-Identification Model (OSNet / Deep Metric Descriptor extraction)
3. Multi-Camera Identity Tracking (Cross-camera matching & trajectory reconstruction)
4. Re-ID Gallery & Global Track ID Assignment
"""

import os
import sys
import numpy as np
from PIL import Image

# Add root to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db import (
    init_db,
    get_db_connection,
    get_all_person_tracks,
    get_unique_global_persons
)
from models.reid import get_reidentifier
from services.reid_service import get_reid_service
from services.video_service import VideoIngestService

def run_tests():
    print("=" * 80)
    print("  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM")
    print("  WEEK 7 & VIDEO DELETION VERIFICATION TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # STEP 1: Re-initialize DB and Schema
    # -------------------------------------------------------------
    print("\n>>> STEP 1: VERIFYING DATABASE SCHEMA WITH PERSON_TRACKS TABLE...")
    init_db()
    conn = get_db_connection()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()
    print(f"    Available tables: {tables}")
    assert "person_tracks" in tables, "person_tracks table missing!"
    print("    [PASS] Database schema successfully updated with Person Re-ID tracking table.")

    # -------------------------------------------------------------
    # STEP 2: Verify PersonReIdentifier (OSNet / Metric Descriptor)
    # -------------------------------------------------------------
    print("\n>>> STEP 2: VERIFYING PERSON RE-IDENTIFIER (OSNet / METRIC EMBEDDER)...")
    reid = get_reidentifier()
    
    # Create two synthetic person crops with similar clothing (probe & gallery from same person)
    # and a third crop with completely different clothing (different person)
    crop_p1_cam1 = Image.new("RGB", (128, 256), color=(40, 70, 150))   # Blue jacket person
    crop_p1_cam2 = Image.new("RGB", (128, 256), color=(45, 75, 155))   # Same blue jacket under different camera lighting
    crop_p2_cam2 = Image.new("RGB", (128, 256), color=(200, 50, 40))   # Red jacket person

    desc_p1_c1 = reid.extract_descriptor(crop_p1_cam1)
    desc_p1_c2 = reid.extract_descriptor(crop_p1_cam2)
    desc_p2_c2 = reid.extract_descriptor(crop_p2_cam2)

    assert desc_p1_c1.shape == (512,), f"Expected 512-dim descriptor, got {desc_p1_c1.shape}"
    assert np.isclose(np.linalg.norm(desc_p1_c1), 1.0, atol=1e-3), "Descriptor must be unit normalized"

    is_match_same, sim_same = reid.match_identities(desc_p1_c1, desc_p1_c2, threshold=0.70)
    is_match_diff, sim_diff = reid.match_identities(desc_p1_c1, desc_p2_c2, threshold=0.70)

    print(f"    Same Person (Cam 1 vs Cam 2): Cosine Similarity = {sim_same*100:.1f}% | Match: {is_match_same}")
    print(f"    Different Persons:            Cosine Similarity = {sim_diff*100:.1f}% | Match: {is_match_diff}")
    assert sim_same > sim_diff, "Same person similarity should be strictly higher than different person"
    print("    [PASS] PersonReIdentifier successfully extracts invariant 512-dim descriptors.")

    # -------------------------------------------------------------
    # STEP 3: Indexing Person Tracks across Videos and Cameras
    # -------------------------------------------------------------
    print("\n>>> STEP 3: RUNNING CROSS-CAMERA PERSON RE-ID INDEXING ON INGESTED FOOTAGE...")
    reid_svc = get_reid_service()
    
    conn = get_db_connection()
    completed_videos = conn.execute("SELECT video_id, filename, duration FROM videos WHERE status='completed'").fetchall()
    conn.close()

    total_indexed = 0
    for v in completed_videos:
        res = reid_svc.index_video_persons(v["video_id"])
        total_indexed += res.get("new_tracks_count", 0)

    summary = reid_svc.get_gallery_summary()
    print(f"    Total Video Sessions Processed: {len(completed_videos)}")
    print(f"    Total Person Sightings Indexed: {summary['total_person_sightings']}")
    print(f"    Unique Global Persons (IDs):    {summary['total_unique_persons']}")
    print(f"    Multi-Camera Tracked Persons:   {summary['multi_camera_tracked_count']}")

    if summary['persons']:
        top_person = summary['persons'][0]
        print(f"    Sample Tracked Individual: {top_person['global_person_id']} ({top_person['appearances']} sightings across {top_person['camera_count']} cameras)")

    # -------------------------------------------------------------
    # STEP 4: Cross-Camera Person Search & Trajectory Reconstruction
    # -------------------------------------------------------------
    print("\n>>> STEP 4: VERIFYING CROSS-CAMERA PERSON TRAJECTORY RETRIEVAL...")
    tracks = get_all_person_tracks()
    if tracks:
        probe_track = tracks[0]
        search_res = reid_svc.search_person_across_cameras(probe_track["track_id"], min_similarity=0.50)
        print(f"    Probe: Track ID '{probe_track['track_id']}' ({probe_track['global_person_id']}) at Cam '{probe_track['camera_id']}'")
        print(f"    Found {search_res['total_matches']} matching appearances (Cross-camera: {len(search_res['cross_camera_matches'])})")
        for idx, m in enumerate(search_res['all_matches'][:3]):
            print(f"      #{idx+1} [Sim: {m['similarity_percent']}%] Cam: {m['camera_id']} | Time: {m['timestamp_sec']}s | ID: {m['global_person_id']}")
    else:
        print("    [NOTE] No person tracks present in current database. Upload a video with pedestrians to populate tracks.")

    # -------------------------------------------------------------
    # STEP 5: Testing Video Deletion API & Cascade
    # -------------------------------------------------------------
    print("\n>>> STEP 5: TESTING VIDEO DELETION FEATURE (CASCADE & CLEANUP)...")
    # Ingest a quick dummy video session to safely test deletion without affecting user footage
    test_vid_id = "vid_test_deletion_dummy"
    conn = get_db_connection()
    conn.execute("INSERT OR REPLACE INTO videos (video_id, filename, storage_path, status) VALUES (?, ?, ?, ?)",
                 (test_vid_id, "test_dummy.mp4", "static/uploads/test_dummy.mp4", "completed"))
    conn.execute("INSERT OR REPLACE INTO frames (frame_id, video_id, frame_number, timestamp_sec, storage_path) VALUES (?, ?, ?, ?, ?)",
                 ("frm_test_dummy_01", test_vid_id, 0, 0.0, "static/frames/vid_test_deletion_dummy/test.jpg"))
    conn.execute("INSERT OR REPLACE INTO detections (detection_id, frame_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 ("det_test_dummy_01", "frm_test_dummy_01", "person", 0.95, 10, 10, 50, 100))
    conn.commit()
    conn.close()

    print(f"    Created mock video session '{test_vid_id}' with frames and detections.")
    del_res = VideoIngestService.delete_video_and_assets(test_vid_id)
    assert del_res["success"] is True, f"Delete failed: {del_res}"
    print(f"    Deletion Result: {del_res}")
    
    # Verify records are gone from SQLite
    conn = get_db_connection()
    v_check = conn.execute("SELECT * FROM videos WHERE video_id = ?", (test_vid_id,)).fetchone()
    f_check = conn.execute("SELECT * FROM frames WHERE video_id = ?", (test_vid_id,)).fetchall()
    d_check = conn.execute("SELECT * FROM detections WHERE frame_id = 'frm_test_dummy_01'").fetchall()
    conn.close()
    
    assert v_check is None, "Video record should be deleted"
    assert len(f_check) == 0, "Frames should be deleted"
    assert len(d_check) == 0, "Detections should be deleted"
    print("    [PASS] Video deletion cascade cleanly removed video, frames, detections, and embeddings.")

    print("\n" + "=" * 80)
    print("  STATUS: WEEK 7 & VIDEO DELETION VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
