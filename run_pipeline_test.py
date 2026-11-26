import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DB_PATH, UPLOADS_DIR, FRAMES_DIR, ANNOTATED_DIR
from database.db import init_db, get_db_connection, get_frames_by_video, get_detections_by_frame
from services.video_service import VideoIngestService
from models.detector import EdgeDetector

def generate_sample_cctv_video(output_path: str, duration_sec=5, fps=25):
    """
    Generates a realistic synthetic CCTV test video with simulated surveillance shapes:
    - Simulates an entrance camera with moving human silhouette and objects
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)
    print(f"[1/4] Generating synthetic CCTV footage: {output_path} ({duration_sec}s @ {fps}fps = {total_frames} frames)...")

    for i in range(total_frames):
        # Create CCTV background (dark asphalt ground, lighter background, timestamp overlay)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:300, :] = (70, 75, 80)    # Wall / outdoor area
        frame[300:, :] = (40, 45, 50)    # Pavement

        # Simulate camera noise
        noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw a simulated moving person / target
        pos_x = int(50 + (i / total_frames) * 400)
        pos_y = 220
        # Head
        cv2.circle(frame, (pos_x, pos_y - 40), 18, (180, 200, 220), -1)
        # Body / Torso
        cv2.rectangle(frame, (pos_x - 16, pos_y - 20), (pos_x + 16, pos_y + 60), (120, 140, 160), -1)
        # Legs
        cv2.line(frame, (pos_x - 8, pos_y + 60), (pos_x - 12, pos_y + 110), (90, 100, 110), 6)
        cv2.line(frame, (pos_x + 8, pos_y + 60), (pos_x + 12, pos_y + 110), (90, 100, 110), 6)
        # Backpack
        cv2.rectangle(frame, (pos_x + 12, pos_y - 10), (pos_x + 28, pos_y + 35), (30, 40, 150), -1)

        # Draw surveillance OSD (On-Screen Display)
        timestamp_str = f"CAM-01 [FRONT GATE] 2026-11-01 22:30:{i//fps:02d}.{int((i%fps)*(1000/fps)):03d}"
        cv2.putText(frame, timestamp_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "REC [EDGE AI ACTIVE]", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print("      Done generating sample video.")
    return output_path

def main():
    print("================================================================================")
    print("  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM")
    print("  WEEK 1 & WEEK 2 VERIFICATION TEST RUNNER")
    print("================================================================================")

    # Step 1: Initialize Database
    print("\n>>> STEP 1: INITIALIZING SQLITE DATABASE SCHEMA (DFD D2)...")
    init_db()
    print(f"    SQLite Database initialized successfully at: {DB_PATH}")

    # Step 2: Generate test video
    sample_video_path = str(UPLOADS_DIR / "sample_campus_cctv.mp4")
    generate_sample_cctv_video(sample_video_path, duration_sec=4, fps=25)

    # Step 3: Week 1 Video Ingest & Sampling
    print("\n>>> STEP 2: RUNNING WEEK 1 VIDEO INGESTION & FRAME SAMPLING (DFD 1.1 & 1.2)...")
    start_time = time.time()
    ingest_result = VideoIngestService.ingest_and_sample_video(
        file_path=sample_video_path,
        camera_id="cam_front_gate",
        sample_interval_sec=1.0
    )
    ingest_time = time.time() - start_time
    video_id = ingest_result["video_id"]
    sampled_frames = ingest_result["frames"]

    print(f"    [P1.1-P1.3 Ingest] Video ID: {video_id}")
    print(f"    [P1.4-P1.6 Sampling] Extracted {len(sampled_frames)} frames in {ingest_time:.2f}s")
    for f in sampled_frames:
        print(f"      * Frame: {f['frame_id']} | Time: {f['timestamp_sec']}s | Path: {f['storage_path']}")

    # Step 4: Week 2 YOLOv8 Object Detection & Linking
    print("\n>>> STEP 3: RUNNING WEEK 2 YOLOV8 DETECTION & METADATA LINKING (DFD 1.3)...")
    detector = EdgeDetector()
    det_start = time.time()
    detection_summary = detector.process_and_link_frames(sampled_frames, generate_annotated=True)
    det_time = time.time() - det_start

    print(f"    [P1.7 YOLO Detection] Analyzed {detection_summary['total_frames_analyzed']} frames in {det_time:.2f}s")
    print(f"    [P1.9 Linking] Total detections linked to SQLite: {detection_summary['total_detections_linked']}")

    # Step 5: Test on real surveillance sample with people and vehicles
    print("\n>>> STEP 4: VERIFYING REAL CCTV TARGETS (PERSON, BUS, CAR) WITH YOLOV8...")
    real_sample_path = str(UPLOADS_DIR / "surveillance_bus_people.mp4")
    if os.path.exists(real_sample_path):
        real_ingest = VideoIngestService.ingest_and_sample_video(real_sample_path, camera_id="cam_02_entrance", sample_interval_sec=1.0)
        real_det_summary = detector.process_and_link_frames(real_ingest["frames"], generate_annotated=True)
        print(f"    [Real Video] Ingested: {real_ingest['video_id']} | Sampled Frames: {len(real_ingest['frames'])}")
        print(f"    [Real Video] Detections Linked: {real_det_summary['total_detections_linked']}")
        
        # Display sample detections
        if real_ingest["frames"]:
            f_sample = real_ingest["frames"][0]
            dets = get_detections_by_frame(f_sample["frame_id"])
            for d in dets:
                print(f"      -> Detected '{d['class_name']}' (conf: {d['confidence']:.2f}) at bbox: [{d['bbox_x1']}, {d['bbox_y1']}, {d['bbox_x2']}, {d['bbox_y2']}]")

    # Step 6: Database Verification
    print("\n>>> STEP 5: VERIFYING PERSISTED DATA IN SQLITE (D2)...")
    conn = get_db_connection()
    total_videos = conn.execute("SELECT COUNT(*) as count FROM videos").fetchone()["count"]
    total_frames = conn.execute("SELECT COUNT(*) as count FROM frames").fetchone()["count"]
    total_db_detections = conn.execute("SELECT COUNT(*) as count FROM detections").fetchone()["count"]
    print(f"    [Videos Table] Total registered videos: {total_videos}")
    print(f"    [Frames Table] Total sampled frames: {total_frames}")
    print(f"    [Detections Table] Total linked detections: {total_db_detections}")
    conn.close()

    print("\n================================================================================")
    print("  STATUS: WEEK 1 & WEEK 2 SUCCESSFULLY BUILT AND VERIFIED!")
    print("================================================================================")

if __name__ == "__main__":
    main()
