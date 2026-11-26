import os
import cv2
import json
import uuid
import numpy as np
from pathlib import Path
from PIL import Image

from config import BASE_DIR, CROPS_DIR
from database.db import (
    get_frames_by_video,
    get_detections_by_frame,
    insert_person_track,
    get_all_person_tracks,
    get_person_track_by_id,
    get_tracks_by_global_id,
    get_unique_global_persons
)
from models.reid import get_reidentifier

class ReIDService:
    """
    Week 7: Multi-Camera Person Re-Identification & Cross-Camera Tracking Service.
    Coordinates cropping, descriptor extraction, global identity assignment,
    and cross-camera trajectory reconstruction.
    """

    def __init__(self):
        self.reidentifier = get_reidentifier()
        self.descriptors_cache = {}  # track_id -> 512-dim np.ndarray
        self.descriptors_file = BASE_DIR / "database" / "reid_descriptors.npy"
        self.mapping_file = BASE_DIR / "database" / "reid_mapping.json"
        self._load_cached_descriptors()

    def _load_cached_descriptors(self):
        """Loads persisted Re-ID descriptors and track mappings."""
        if self.descriptors_file.exists() and self.mapping_file.exists():
            try:
                vecs = np.load(str(self.descriptors_file))
                with open(self.mapping_file, "r") as f:
                    track_ids = json.load(f)
                for tid, vec in zip(track_ids, vecs):
                    self.descriptors_cache[tid] = vec
            except Exception as e:
                print(f"Warning: Failed to load Re-ID descriptor cache: {e}")

    def _save_cached_descriptors(self):
        """Saves Re-ID descriptors and track mappings to disk."""
        if not self.descriptors_cache:
            return
        track_ids = list(self.descriptors_cache.keys())
        vecs = np.array([self.descriptors_cache[tid] for tid in track_ids], dtype=np.float32)
        np.save(str(self.descriptors_file), vecs)
        with open(self.mapping_file, "w") as f:
            json.dump(track_ids, f)

    def index_video_persons(self, video_id: str, similarity_threshold: float = 0.72) -> dict:
        """
        Crops all detected persons in the video's frames, extracts Re-ID descriptors,
        assigns global identities, and saves tracks in SQLite.
        """
        frames = get_frames_by_video(video_id)
        if not frames:
            return {"success": False, "error": f"No frames found for video {video_id}", "indexed_count": 0}

        video_crops_dir = CROPS_DIR / video_id
        os.makedirs(video_crops_dir, exist_ok=True)

        new_tracks = 0
        existing_tracks = get_all_person_tracks()
        
        # Build gallery of known persons: global_person_id -> list of descriptors
        gallery = {}
        for tr in existing_tracks:
            gid = tr["global_person_id"]
            tid = tr["track_id"]
            if tid in self.descriptors_cache:
                gallery.setdefault(gid, []).append(self.descriptors_cache[tid])

        next_person_idx = len(gallery) + 1

        for frame_record in frames:
            frame_id = frame_record["frame_id"]
            frame_path = frame_record["storage_path"]
            camera_id = frame_record.get("camera_id", "cam_01")
            timestamp_sec = frame_record["timestamp_sec"]

            if not os.path.exists(frame_path):
                continue

            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                continue

            h, w, _ = frame_img.shape
            dets = get_detections_by_frame(frame_id)

            for d in dets:
                if d["class_name"].lower() != "person":
                    continue

                x1 = max(0, int(d["bbox_x1"]))
                y1 = max(0, int(d["bbox_y1"]))
                x2 = min(w, int(d["bbox_x2"]))
                y2 = min(h, int(d["bbox_y2"]))

                if (x2 - x1) < 20 or (y2 - y1) < 30:
                    continue  # Ignore tiny or distant boxes

                crop = frame_img[y1:y2, x1:x2]
                track_id = f"trk_{uuid.uuid4().hex[:12]}"
                crop_filename = f"crop_{track_id}_{d['detection_id'][:8]}.jpg"
                crop_path = video_crops_dir / crop_filename

                cv2.imwrite(str(crop_path), crop)

                # Extract 512-dim normalized feature vector
                descriptor = self.reidentifier.extract_descriptor(crop)
                self.descriptors_cache[track_id] = descriptor

                # Match with gallery to determine global_person_id
                assigned_gid = None
                best_sim = -1.0

                for gid, descs in gallery.items():
                    for g_desc in descs:
                        sim = self.reidentifier.compute_similarity(descriptor, g_desc)
                        if sim > best_sim:
                            best_sim = sim
                            if sim >= similarity_threshold:
                                assigned_gid = gid

                if not assigned_gid:
                    assigned_gid = f"PERSON_{next_person_idx:03d}"
                    next_person_idx += 1

                gallery.setdefault(assigned_gid, []).append(descriptor)

                # Relative path for web serving
                rel_crop_path = os.path.relpath(str(crop_path), BASE_DIR).replace("\\", "/")

                insert_person_track(
                    track_id=track_id,
                    global_person_id=assigned_gid,
                    detection_id=d["detection_id"],
                    frame_id=frame_id,
                    video_id=video_id,
                    camera_id=camera_id,
                    timestamp_sec=timestamp_sec,
                    crop_path=f"/{rel_crop_path}",
                    confidence=float(d["confidence"])
                )
                new_tracks += 1

        self._save_cached_descriptors()

        return {
            "success": True,
            "video_id": video_id,
            "new_tracks_count": new_tracks,
            "total_global_persons": len(gallery)
        }

    def search_person_across_cameras(self, probe_track_id: str, min_similarity: float = 0.55) -> dict:
        """
        Cross-camera person retrieval:
        Takes a probe person track and matches them across other camera feeds.
        Returns ranked trajectory with similarity scores and timestamps.
        """
        probe_track = get_person_track_by_id(probe_track_id)
        if not probe_track:
            return {"success": False, "error": f"Track {probe_track_id} not found", "matches": []}

        probe_desc = self.descriptors_cache.get(probe_track_id)
        if probe_desc is None:
            # Try extracting from crop on disk
            crop_abs = BASE_DIR / probe_track["crop_path"].lstrip("/")
            if crop_abs.exists():
                probe_desc = self.reidentifier.extract_descriptor(str(crop_abs))
                self.descriptors_cache[probe_track_id] = probe_desc

        if probe_desc is None:
            return {"success": False, "error": "Could not obtain feature descriptor for probe", "matches": []}

        all_tracks = get_all_person_tracks()
        matches = []

        for tr in all_tracks:
            tid = tr["track_id"]
            if tid == probe_track_id:
                continue

            desc = self.descriptors_cache.get(tid)
            if desc is None:
                crop_abs = BASE_DIR / tr["crop_path"].lstrip("/")
                if crop_abs.exists():
                    desc = self.reidentifier.extract_descriptor(str(crop_abs))
                    self.descriptors_cache[tid] = desc

            if desc is not None:
                sim = self.reidentifier.compute_similarity(probe_desc, desc)
                if sim >= min_similarity:
                    tr_copy = dict(tr)
                    tr_copy["similarity_score"] = round(sim, 4)
                    tr_copy["similarity_percent"] = int(sim * 100)
                    tr_copy["is_cross_camera"] = (tr["camera_id"] != probe_track["camera_id"])
                    matches.append(tr_copy)

        # Sort matches by similarity score descending
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)

        return {
            "success": True,
            "probe": probe_track,
            "total_matches": len(matches),
            "cross_camera_matches": [m for m in matches if m["is_cross_camera"]],
            "all_matches": matches
        }

    def get_gallery_summary(self) -> dict:
        """Returns gallery overview of tracked identities and multi-camera appearances."""
        unique_persons = get_unique_global_persons()
        all_tracks = get_all_person_tracks()
        
        multi_cam_count = sum(1 for p in unique_persons if p["camera_count"] > 1)
        
        return {
            "total_unique_persons": len(unique_persons),
            "multi_camera_tracked_count": multi_cam_count,
            "total_person_sightings": len(all_tracks),
            "persons": unique_persons,
            "recent_tracks": all_tracks[:20]
        }

# Global singleton
_reid_svc_instance = None

def get_reid_service() -> ReIDService:
    global _reid_svc_instance
    if _reid_svc_instance is None:
        _reid_svc_instance = ReIDService()
    return _reid_svc_instance
