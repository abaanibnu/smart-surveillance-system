import os
import uuid
import numpy as np
from PIL import Image
from pathlib import Path
from models.embedder import ClipEmbedder
from models.faiss_index import FAISSVectorIndex
from services.ontology_service import OntologyEngine
from services.speech_service import VoiceQueryHandler
from database.db import (
    get_db_connection,
    insert_embeddings_batch,
    get_unembedded_frames,
    get_frame_details_for_search,
    get_all_frames
)

class SearchService:
    """
    Implements:
    - Week 3: Frame visual embedding generation & indexing (P1.8 - P1.11)
    - Week 4: Natural language query embedding & FAISS retrieval (P2.3 - P2.6)
    - Week 5: Ontology concept expansion (P2.2) via Protégé surveillance.owl & RDFLib
    - Week 6: Whisper Speech-to-Text voice querying (DFD 1.5 & Sequence Diagram 5.14)
    - Precision Attribute & Personal Eyewear/Glasses Verification
    """

    EYEWEAR_KEYWORDS = {"glasses", "sunglasses", "eyeglasses", "spectacles", "eyewear", "shades", "specs"}

    ONTOLOGY_YOLO_MAP = {
        "vehicle": ["car", "bus", "motorcycle", "truck", "bicycle"],
        "car": ["car"],
        "cars": ["car"],
        "automobile": ["car"],
        "sedan": ["car"],
        "suv": ["car"],
        "bus": ["bus"],
        "buses": ["bus"],
        "shuttle": ["bus"],
        "coach": ["bus"],
        "transit": ["bus"],
        "motorcycle": ["motorcycle"],
        "motorcycles": ["motorcycle"],
        "motorbike": ["motorcycle"],
        "scooter": ["motorcycle"],
        "bicycle": ["bicycle"],
        "bike": ["bicycle", "motorcycle"],
        "truck": ["truck"],
        "trucks": ["truck"],
        "van": ["truck", "car"],
        "lorry": ["truck"],
        "person": ["person"],
        "people": ["person"],
        "human": ["person"],
        "pedestrian": ["person"],
        "man": ["person"],
        "woman": ["person"],
        "guy": ["person"],
        "boy": ["person"],
        "girl": ["person"],
        "backpack": ["backpack"],
        "bag": ["backpack", "handbag"],
        "rucksack": ["backpack"],
        "knapsack": ["backpack"],
        "handbag": ["handbag"],
        "purse": ["handbag"],
        "tote": ["handbag"],
        "cell phone": ["cell phone"],
        "phone": ["cell phone"],
        "smartphone": ["cell phone"],
        "mobile": ["cell phone"]
    }

    COLOR_MAP = {
        "black": ("black", "white, light, or bright colorful"),
        "white": ("white", "black or dark"),
        "red": ("red", "black, blue, or white"),
        "blue": ("blue", "black, red, or white"),
        "yellow": ("yellow", "black, dark, or blue"),
        "green": ("green", "black or red"),
        "dark": ("dark colored", "light or white"),
        "light": ("light colored", "dark or black")
    }

    def __init__(self, embedder: ClipEmbedder = None, vector_index: FAISSVectorIndex = None, ontology: OntologyEngine = None, voice: VoiceQueryHandler = None):
        self.embedder = embedder or ClipEmbedder()
        self.vector_index = vector_index or FAISSVectorIndex()
        self.ontology_engine = ontology or OntologyEngine()
        self._voice_handler = voice
        self._crop_cache = {}  # crop_path -> (full_emb, head_emb)

    def _get_crop_embeddings(self, crop_path: str):
        """Retrieves or computes and caches 512-dim visual embeddings for a person crop."""
        if crop_path in self._crop_cache:
            return self._crop_cache[crop_path]
        
        cfile = Path(crop_path.lstrip("/"))
        if not cfile.exists():
            return None, None

        try:
            cimg = Image.open(cfile)
            w, h = cimg.size
            full_emb = self.embedder.embed_image(cimg)
            head_crop = cimg.crop((0, 0, w, int(h * 0.45)))
            head_emb = self.embedder.embed_image(head_crop)
            self._crop_cache[crop_path] = (full_emb, head_emb)
            return full_emb, head_emb
        except Exception:
            return None, None

    @property
    def voice_handler(self):
        if self._voice_handler is None:
            self._voice_handler = VoiceQueryHandler()
        return self._voice_handler

    def index_frames(self, frames_list: list):
        """
        P1.8 - P1.11: Embeds a list of sampled frames with CLIP and saves to FAISS & SQLite.
        Also creates multi-scale visual embeddings for any detected person crops in the frame.
        """
        if not frames_list:
            return 0

        embeddings_db_batch = []
        indexed_count = 0

        for f in frames_list:
            frame_id = f["frame_id"]
            storage_path = f["storage_path"]

            if not os.path.exists(storage_path):
                continue

            try:
                # P1.8: CLIP image embedding (Global Scene)
                vec_512 = self.embedder.embed_image(storage_path)
                vector_id = self.vector_index.add_vector(frame_id, vec_512)
                
                emb_id = f"emb_{uuid.uuid4().hex[:12]}"
                embeddings_db_batch.append((emb_id, frame_id, vector_id, 1))
                indexed_count += 1

                # Multi-Scale Visual Indexing: Index person crops and upper-body crops
                try:
                    conn = get_db_connection()
                    tracks = conn.execute("SELECT track_id, crop_path FROM person_tracks WHERE frame_id = ?", (frame_id,)).fetchall()
                    conn.close()
                    for idx, t in enumerate(tracks):
                        c_path = Path(t['crop_path'].lstrip('/'))
                        if c_path.exists():
                            c_img = Image.open(c_path)
                            w, h = c_img.size
                            # 1. Full body crop
                            v_crop = self.embedder.embed_image(c_img)
                            self.vector_index.add_vector(f"{frame_id}#crop_{idx}", v_crop)
                            # 2. Upper body / head crop (top 45%)
                            head_img = c_img.crop((0, 0, w, int(h * 0.45)))
                            v_head = self.embedder.embed_image(head_img)
                            self.vector_index.add_vector(f"{frame_id}#head_{idx}", v_head)
                except Exception as crop_err:
                    pass

            except Exception as e:
                print(f"Error embedding frame {frame_id}: {e}")

        if embeddings_db_batch:
            insert_embeddings_batch(embeddings_db_batch)
            self.vector_index.save()

        return indexed_count

    def index_pending_frames(self) -> int:
        """Indexes all frames in the database that do not yet have embeddings."""
        pending = get_unembedded_frames()
        return self.index_frames(pending)

    def search(self, query_text: str, top_k: int = 12, min_similarity: float = 0.15, use_ontology: bool = True) -> dict:
        """
        Generalized Multi-Modal Precision Semantic Search (DFD Level 1.5 - 1.7):
        - Ontology Entity & Concept Grounding
        - Cross-Modal YOLOv8 Detection Class Alignment
        - Multi-Scale Person & Object Crop Verification
        - Differential Visual Attribute Contrast (Colors, Eyewear, Wearables)
        - Calibrated Confidence Scoring & Dynamic Semantic Badges
        """
        if not query_text or not query_text.strip():
            return {
                "query": "",
                "composite_query": "",
                "expanded_terms": [],
                "detected_concepts": {},
                "total_matches": 0,
                "results": []
            }

        clean_query = query_text.strip().lower()
        query_tokens = set(clean_query.split())

        # 1. Query Concept & Target Class Extraction
        target_classes = set()
        for kw, yclasses in self.ONTOLOGY_YOLO_MAP.items():
            if kw in clean_query:
                target_classes.update(yclasses)

        detected_colors = [c for c in self.COLOR_MAP if c in clean_query]
        is_eyewear_query = bool(query_tokens & self.EYEWEAR_KEYWORDS or any(k in clean_query for k in self.EYEWEAR_KEYWORDS))
        is_person_query = bool("person" in target_classes or is_eyewear_query or detected_colors or any(act in clean_query for act in ("walking", "running", "standing", "sitting")))

        # 2. Ontology Expansion
        if use_ontology:
            expansion = self.ontology_engine.expand_full_query(query_text.strip())
            composite_query = expansion["composite_query"]
            detected_concepts = expansion["detected_concepts"]
            expanded_terms = expansion["expanded_terms"]
        else:
            composite_query = query_text.strip()
            detected_concepts = {}
            expanded_terms = []

        # 3. Focused Prompt Ensemble for Semantic Vector
        prompts = [
            clean_query,
            f"a surveillance camera photo of {clean_query}",
            f"a security video frame showing {clean_query}"
        ]
        if composite_query and composite_query.lower() != clean_query:
            prompts.append(f"a surveillance recording showing {composite_query}")

        prompt_vecs = [self.embedder.embed_text(p) for p in prompts]
        query_vector = np.mean(prompt_vecs, axis=0)
        query_vector /= np.linalg.norm(query_vector)

        # Attribute contrast vectors (Positive vs Negative)
        pos_attr_vec, neg_attr_vec = None, None
        if is_eyewear_query:
            pos_attr_vec = self.embedder.embed_text("a person face wearing glasses or sunglasses")
            neg_attr_vec = self.embedder.embed_text("a bare human face without glasses or eyewear")
        elif detected_colors and is_person_query:
            col = detected_colors[0]
            pos_attr_vec = self.embedder.embed_text(f"a person wearing {col} clothes")
            neg_attr_vec = self.embedder.embed_text(f"a person wearing {self.COLOR_MAP[col][1]} clothes")

        # 4. Multi-Scale Candidate Sourcing (FAISS Vectors + SQLite Detections)
        raw_matches = self.vector_index.search(query_vector, top_k=top_k * 6, min_score=0.10)
        candidate_map = {}
        for m in raw_matches:
            raw_fid = m["frame_id"]
            base_fid = raw_fid.split("#")[0]
            sim_score = m["similarity"]
            if base_fid not in candidate_map or sim_score > candidate_map[base_fid]:
                candidate_map[base_fid] = sim_score

        conn = get_db_connection()
        if target_classes:
            placeholders = ",".join("?" for _ in target_classes)
            rows = conn.execute(
                f"SELECT DISTINCT frame_id FROM detections WHERE class_name IN ({placeholders}) LIMIT 40",
                list(target_classes)
            ).fetchall()
            for r in rows:
                fid = r["frame_id"]
                if fid not in candidate_map:
                    if fid in self.vector_index.frame_to_id:
                        v_idx = self.vector_index.frame_to_id[fid]
                        fvec = self.vector_index.vectors[v_idx]
                        candidate_map[fid] = float(np.dot(query_vector, fvec))
                    else:
                        candidate_map[fid] = 0.20

        # 5. Precision Cross-Modal Verification & Scoring
        ranked_results = []
        for base_frame_id, sim_base in candidate_map.items():
            frame_data = get_frame_details_for_search(base_frame_id)
            if not frame_data:
                continue

            dets = frame_data.get("detections", [])
            det_classes = [d["class_name"] for d in dets]
            tracks = conn.execute(
                "SELECT track_id, crop_path, global_person_id FROM person_tracks WHERE frame_id = ?",
                (base_frame_id,)
            ).fetchall()

            matched_badge = None
            class_score = 0.0
            crop_bonus = 0.0
            best_crop_path = None
            glasses_detected = False

            # A. Evaluate target class alignment
            if target_classes:
                matching_dets = [d for d in dets if d["class_name"] in target_classes]
                if matching_dets:
                    top_det = matching_dets[0]
                    conf = top_det["confidence"]
                    cname = top_det["class_name"]
                    class_score = 0.28 + 0.10 * conf
                    matched_badge = f"{cname.capitalize()} Confirmed ({int(conf * 100)}% conf)"
                else:
                    # Explicit class missing in this frame
                    class_score = -0.18

            # B. Evaluate person crops & attributes
            if is_person_query and tracks:
                best_person_score = 0.0
                best_pid = ""
                for t in tracks:
                    full_emb, head_emb = self._get_crop_embeddings(t["crop_path"])
                    if full_emb is not None:
                        psim = float(np.dot(query_vector, full_emb))
                        
                        if is_eyewear_query and head_emb is not None:
                            diff = float(np.dot(pos_attr_vec, head_emb)) - float(np.dot(neg_attr_vec, head_emb))
                            if diff > 0.005:
                                psim += 0.35 + diff * 3.0
                                matched_badge = f"Person ({t['global_person_id']}) Wearing Glasses"
                                glasses_detected = True
                                best_crop_path = t["crop_path"]
                        elif detected_colors:
                            diff = float(np.dot(pos_attr_vec, full_emb)) - float(np.dot(neg_attr_vec, full_emb))
                            if diff > 0:
                                psim += 0.25 + diff * 2.0
                                matched_badge = f"Person in {detected_colors[0].capitalize()} ({t['global_person_id']})"
                                best_crop_path = t["crop_path"]

                        if psim > best_person_score:
                            best_person_score = psim
                            best_pid = t["global_person_id"]
                            if not best_crop_path:
                                best_crop_path = t["crop_path"]

                if best_person_score > 0:
                    crop_bonus = max(0.0, best_person_score - 0.20)
                    if not matched_badge:
                        matched_badge = f"Person Verified ({best_pid})"

            # C. Calibrated confidence formula
            raw_score = sim_base + class_score + crop_bonus
            if class_score > 0 or crop_bonus > 0.15:
                calibrated_sim = min(0.92, max(0.60, raw_score * 1.55))
            else:
                calibrated_sim = max(0.08, min(0.35, raw_score * 0.80))

            final_sim = round(calibrated_sim, 3)

            from config import BASE_DIR, ANNOTATED_DIR
            frame_name = Path(frame_data["storage_path"]).name
            annotated_name = f"annotated_{frame_name}"
            annotated_file = ANNOTATED_DIR / annotated_name
            rel_frame_path = os.path.relpath(frame_data["storage_path"], BASE_DIR).replace("\\", "/")
            annotated_url = f"/static/annotated/{annotated_name}" if annotated_file.exists() else None

            ranked_results.append({
                "frame_id": frame_data["frame_id"],
                "video_id": frame_data["video_id"],
                "video_name": frame_data["filename"],
                "camera_id": frame_data["camera_id"],
                "timestamp_sec": frame_data["timestamp_sec"],
                "similarity_score": final_sim,
                "similarity_percent": int(final_sim * 100),
                "frame_url": f"/{rel_frame_path}",
                "annotated_url": annotated_url,
                "detections": frame_data["detections"],
                "matched_attribute": matched_badge,
                "glasses_detected": glasses_detected,
                "person_crop_url": best_crop_path
            })

        conn.close()

        # Sort by final similarity descending and filter by min_similarity
        ranked_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        final_results = [r for r in ranked_results if r["similarity_score"] >= min_similarity][:top_k]

        return {
            "query": query_text.strip(),
            "composite_query": composite_query,
            "expanded_terms": expanded_terms,
            "detected_concepts": detected_concepts,
            "total_matches": len(final_results),
            "results": final_results
        }

    def search_voice(self, audio_path: str, top_k: int = 12, min_similarity: float = 0.15) -> dict:
        """
        Week 6: Voice-driven search pipeline (Sequence Diagram 5.14).
        Transcribes voice with Whisper, runs ontology expansion and semantic retrieval.
        """
        transcription_res = self.voice_handler.transcribe_audio(audio_path)
        if not transcription_res.get("success"):
            return {
                "success": False,
                "error": transcription_res.get("error", "Failed to transcribe audio"),
                "transcription": "",
                "results": []
            }

        transcribed_text = transcription_res["text"]
        search_output = self.search(transcribed_text, top_k=top_k, min_similarity=min_similarity, use_ontology=True)
        search_output["success"] = True
        search_output["transcription"] = transcribed_text
        search_output["language"] = transcription_res.get("language", "en")
        return search_output

# Global singleton
_search_service_instance = None

def get_search_service() -> SearchService:
    global _search_service_instance
    if _search_service_instance is None:
        _search_service_instance = SearchService()
    return _search_service_instance
