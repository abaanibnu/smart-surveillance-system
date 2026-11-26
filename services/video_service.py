import os
import uuid
import cv2
from pathlib import Path
from config import FRAMES_DIR, ALLOWED_EXTENSIONS, DEFAULT_SAMPLE_INTERVAL_SEC
from database.db import insert_video, update_video_status, insert_frames_batch

class VideoIngestService:
    """
    Implements DFD Levels 1.1 and 1.2:
    - P1.1: Video Input (file or camera stream)
    - P1.2: Validation (format, codec, readability, metadata)
    - P1.3: Ingest Record creation
    - P1.4: Read Video
    - P1.5: Frame Sampling at configured time intervals
    - P1.6: Prepare & Save Frames
    """

    @staticmethod
    def validate_video_file(file_path: str):
        """
        Validates file existence, extension, and OpenCV readability.
        Returns: (is_valid: bool, error_msg: str, metadata: dict)
        """
        path = Path(file_path)
        if not path.exists():
            return False, f"File does not exist: {file_path}", {}

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported video format: {path.suffix}. Allowed: {ALLOWED_EXTENSIONS}", {}

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False, "Failed to open video source with OpenCV. Codec may be corrupted or unsupported.", {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps and fps > 0 else 0.0

        ret, test_frame = cap.read()
        cap.release()

        if not ret or test_frame is None:
            return False, "Video opened but no valid frames could be decoded.", {}

        metadata = {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": round(duration, 2),
            "filename": path.name,
            "storage_path": str(path)
        }
        return True, "", metadata

    @classmethod
    def ingest_and_sample_video(cls, file_path: str, camera_id="cam_01", sample_interval_sec=DEFAULT_SAMPLE_INTERVAL_SEC):
        """
        Full Level 1.1 & Level 1.2 pipeline.
        Samples frames at regular intervals and saves them to FRAMES_DIR.
        Records entries into SQLite database.
        """
        is_valid, err, meta = cls.validate_video_file(file_path)
        if not is_valid:
            raise ValueError(f"Video validation failed: {err}")

        video_id = f"vid_{uuid.uuid4().hex[:12]}"
        
        # P1.3: Ingest Record
        insert_video(
            video_id=video_id,
            filename=meta["filename"],
            storage_path=meta["storage_path"],
            duration=meta["duration"],
            fps=meta["fps"],
            total_frames=meta["total_frames"],
            status="processing"
        )

        # P1.4: Read Video
        cap = cv2.VideoCapture(meta["storage_path"])
        fps = meta["fps"] if meta["fps"] > 0 else 25.0
        frame_step = max(1, int(fps * sample_interval_sec))

        frames_db_batch = []
        extracted_frames_info = []

        current_frame_idx = 0
        video_frames_dir = FRAMES_DIR / video_id
        os.makedirs(video_frames_dir, exist_ok=True)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # P1.5: Sample frames at intervals
                if current_frame_idx % frame_step == 0:
                    timestamp_sec = round(current_frame_idx / fps, 2)
                    frame_id = f"frm_{uuid.uuid4().hex[:12]}"
                    frame_filename = f"{frame_id}_{timestamp_sec:.2f}s.jpg"
                    frame_save_path = str(video_frames_dir / frame_filename)

                    # P1.6: Prepare & save frame
                    cv2.imwrite(frame_save_path, frame)

                    frame_record = {
                        "frame_id": frame_id,
                        "video_id": video_id,
                        "camera_id": camera_id,
                        "frame_number": current_frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "storage_path": frame_save_path
                    }
                    extracted_frames_info.append(frame_record)

                    frames_db_batch.append((
                        frame_id,
                        video_id,
                        camera_id,
                        current_frame_idx,
                        timestamp_sec,
                        frame_save_path
                    ))

                current_frame_idx += 1

            cap.release()

            # Batch insert frames into SQLite
            if frames_db_batch:
                insert_frames_batch(frames_db_batch)

            update_video_status(video_id, status="completed")

            return {
                "success": True,
                "video_id": video_id,
                "metadata": meta,
                "sampled_frames_count": len(extracted_frames_info),
                "frames": extracted_frames_info
            }

        except Exception as e:
            cap.release()
            update_video_status(video_id, status="failed")
            raise RuntimeError(f"Error during frame sampling: {str(e)}")

    @classmethod
    def delete_video_and_assets(cls, video_id: str):
        """
        Deletes video and cascades all sampled frames, annotations,
        person crops, detections, embeddings, and FAISS vector mappings.
        """
        import shutil
        from database.db import delete_video_cascade
        from config import ANNOTATED_DIR, FRAMES_DIR, CROPS_DIR

        video_info, frame_ids, stats = delete_video_cascade(video_id)
        if not video_info:
            return {"success": False, "error": f"Video '{video_id}' not found"}

        # 1. Delete extracted frames directory
        video_frames_dir = FRAMES_DIR / video_id
        if video_frames_dir.exists():
            shutil.rmtree(video_frames_dir, ignore_errors=True)

        # 2. Delete annotated frame images
        for fid in frame_ids:
            for p in ANNOTATED_DIR.glob(f"annotated_{fid}*"):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        # 3. Delete person crops directory
        video_crops_dir = CROPS_DIR / video_id
        if video_crops_dir.exists():
            shutil.rmtree(video_crops_dir, ignore_errors=True)

        # 4. Remove vectors from FAISS index
        try:
            from services.search_service import get_search_service
            search_svc = get_search_service()
            search_svc.vector_index.remove_frames(frame_ids)
        except Exception as e:
            print(f"Warning: Failed to update FAISS index on video delete: {e}")

        return {
            "success": True,
            "video_id": video_id,
            "filename": video_info["filename"],
            "deleted_frames_count": stats["deleted_frames_count"],
            "deleted_detections_count": stats["deleted_detections_count"],
            "deleted_embeddings_count": stats["deleted_embeddings_count"]
        }
