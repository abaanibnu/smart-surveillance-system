import os
import json
from pathlib import Path
import numpy as np

class FAISSVectorIndex:
    """
    Implements DFD Levels 1.4 & 1.6:
    - P1.10: Store Vectors (Index visual embeddings)
    - P2.4 - P2.6: Vector Similarity Search & Ranking
    Uses FAISS IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)
    with a vectorized NumPy fallback when FAISS C-bindings are unavailable.
    """

    def __init__(self, dimension=512, index_dir=None):
        self.dimension = dimension
        if index_dir is None:
            from config import BASE_DIR
            self.index_dir = BASE_DIR / "database"
        else:
            self.index_dir = Path(index_dir)
        os.makedirs(self.index_dir, exist_ok=True)

        self.index_file = self.index_dir / "surveillance_vectors.npy"
        self.encrypted_file = self.index_dir / "surveillance_vectors.enc"
        self.mapping_file = self.index_dir / "vector_id_mapping.json"

        self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.id_to_frame = {}  # int_id -> frame_id string
        self.frame_to_id = {}  # frame_id -> int_id

        self._has_faiss = False
        try:
            import faiss
            self._faiss = faiss
            self._faiss_index = faiss.IndexFlatIP(self.dimension)
            self._has_faiss = True
        except ImportError:
            self._faiss = None
            self._faiss_index = None

        self.load()

    @property
    def total_vectors(self) -> int:
        return len(self.vectors)

    def add_vector(self, frame_id: str, vector_512: np.ndarray) -> int:
        """
        Adds a single normalized 512-dim embedding to the vector index.
        Returns the integer vector_id.
        """
        if frame_id in self.frame_to_id:
            return self.frame_to_id[frame_id]

        vec = np.asarray(vector_512, dtype=np.float32).reshape(1, self.dimension)
        # Ensure L2 normalized
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        vector_id = len(self.vectors)
        if len(self.vectors) == 0:
            self.vectors = vec
        else:
            self.vectors = np.vstack([self.vectors, vec])

        self.id_to_frame[str(vector_id)] = frame_id
        self.frame_to_id[frame_id] = vector_id

        if self._has_faiss and self._faiss_index is not None:
            self._faiss_index.add(vec)

        return vector_id

    def add_vectors_batch(self, frame_ids: list, vectors_batch: np.ndarray) -> list:
        """
        Adds a batch of vectors (N, 512).
        Returns a list of integer vector_ids.
        """
        vecs = np.asarray(vectors_batch, dtype=np.float32)
        if len(vecs.shape) == 1:
            vecs = vecs.reshape(1, self.dimension)

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        start_id = len(self.vectors)
        assigned_ids = []

        if len(self.vectors) == 0:
            self.vectors = vecs
        else:
            self.vectors = np.vstack([self.vectors, vecs])

        for i, fid in enumerate(frame_ids):
            vid = start_id + i
            self.id_to_frame[str(vid)] = fid
            self.frame_to_id[fid] = vid
            assigned_ids.append(vid)

        if self._has_faiss and self._faiss_index is not None:
            self._faiss_index.add(vecs)

        return assigned_ids

    def remove_frames(self, frame_ids: list) -> int:
        """
        Removes vectors associated with the given frame_ids,
        rebuilds the index and mappings, and persists to disk.
        """
        remove_set = set(frame_ids)
        if not remove_set or len(self.vectors) == 0:
            return 0

        keep_indices = []
        new_id_to_frame = {}
        new_frame_to_id = {}

        new_idx = 0
        for old_id_str, fid in self.id_to_frame.items():
            if fid not in remove_set:
                keep_indices.append(int(old_id_str))
                new_id_to_frame[str(new_idx)] = fid
                new_frame_to_id[fid] = new_idx
                new_idx += 1

        removed_count = len(self.vectors) - len(keep_indices)
        if len(keep_indices) > 0:
            self.vectors = self.vectors[keep_indices]
        else:
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)

        self.id_to_frame = new_id_to_frame
        self.frame_to_id = new_frame_to_id

        if self._has_faiss and self._faiss_index is not None:
            self._faiss_index.reset()
            if len(self.vectors) > 0:
                self._faiss_index.add(self.vectors)

        self.save()
        return removed_count

    def search(self, query_vector: np.ndarray, top_k=10, min_score=0.15) -> list:
        """
        P2.4 - P2.6: Similarity Search and Ranking.
        Compares query vector with stored vectors using Inner Product (Cosine Similarity).
        Returns list of dicts: [{'vector_id': int, 'frame_id': str, 'similarity': float}, ...]
        """
        if len(self.vectors) == 0:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32).reshape(1, self.dimension)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        if self._has_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q_vec, min(top_k, len(self.vectors)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= min_score:
                    frame_id = self.id_to_frame.get(str(idx), "")
                    results.append({
                        "vector_id": int(idx),
                        "frame_id": frame_id,
                        "similarity": round(float(score), 4)
                    })
            return results

        # Vectorized NumPy cosine similarity: dot product of (1, 512) and (N, 512).T = (N,)
        similarities = np.dot(self.vectors, q_vec.T).flatten()
        top_k = min(top_k, len(similarities))
        
        # Partition / sort top K in descending order
        top_indices = np.argsort(-similarities)[:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                frame_id = self.id_to_frame.get(str(idx), "")
                results.append({
                    "vector_id": int(idx),
                    "frame_id": frame_id,
                    "similarity": round(score, 4)
                })

        return results

    def save(self):
        """Persists vectors and ID mappings to disk with AES-256 encryption at rest."""
        np.save(str(self.index_file), self.vectors)
        with open(self.mapping_file, "w") as f:
            json.dump({
                "id_to_frame": self.id_to_frame,
                "frame_to_id": self.frame_to_id
            }, f, indent=2)

        # AES-256 Encryption at Rest (Week 9)
        try:
            from services.security_service import get_security_manager
            sec_mgr = get_security_manager()
            if len(self.vectors) > 0:
                cipher_bytes = sec_mgr.encrypt_array(self.vectors)
                with open(self.encrypted_file, "wb") as f:
                    f.write(cipher_bytes)
        except Exception as e:
            print(f"Warning: Failed to encrypt vector store with AES-256: {e}")

    def load(self):
        """Loads persisted index from disk, prioritizing AES-256 encrypted storage."""
        loaded = False
        if self.encrypted_file.exists() and self.mapping_file.exists():
            try:
                from services.security_service import get_security_manager
                sec_mgr = get_security_manager()
                with open(self.encrypted_file, "rb") as f:
                    cipher_bytes = f.read()
                self.vectors = sec_mgr.decrypt_array(cipher_bytes)
                with open(self.mapping_file, "r") as f:
                    mapping = json.load(f)
                    self.id_to_frame = mapping.get("id_to_frame", {})
                    self.frame_to_id = mapping.get("frame_to_id", {})
                if self._has_faiss and self._faiss_index is not None and len(self.vectors) > 0:
                    self._faiss_index.reset()
                    self._faiss_index.add(self.vectors)
                loaded = True
            except Exception as e:
                print(f"Warning: Failed to load AES-256 encrypted vector index: {e}")

        # Fallback to plain .npy if not loaded from encrypted file
        if not loaded and self.index_file.exists() and self.mapping_file.exists():
            try:
                self.vectors = np.load(str(self.index_file))
                with open(self.mapping_file, "r") as f:
                    mapping = json.load(f)
                    self.id_to_frame = mapping.get("id_to_frame", {})
                    self.frame_to_id = mapping.get("frame_to_id", {})
                if self._has_faiss and self._faiss_index is not None and len(self.vectors) > 0:
                    self._faiss_index.reset()
                    self._faiss_index.add(self.vectors)
                # Auto-encrypt to .enc for future runs
                self.save()
            except Exception as e:
                print(f"Warning: Failed to load existing vector index: {e}")
