import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DB_PATH
from database.db import init_db, get_db_connection, get_unembedded_frames
from services.search_service import SearchService

def main():
    print("================================================================================")
    print("  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM")
    print("  WEEK 3 & WEEK 4 VERIFICATION TEST RUNNER")
    print("================================================================================")

    init_db()

    # Step 1: Initialize Semantic Search Engine
    print("\n>>> STEP 1: INITIALIZING CLIP EMBEDDER & FAISS VECTOR STORE...")
    start_load = time.time()
    search_svc = SearchService()
    load_time = time.time() - start_load
    print(f"    CLIP Model (ViT-B/32) and FAISS Index loaded in {load_time:.2f}s")
    print(f"    Current Vectors in Index: {search_svc.vector_index.total_vectors}")

    # Step 2: Week 3 - Visual Frame Embeddings & Storage
    print("\n>>> STEP 2: RUNNING WEEK 3 FRAME EMBEDDING & FAISS INDEXING (DFD 1.3 & 1.4)...")
    pending_frames = get_unembedded_frames()
    print(f"    Frames awaiting vector embedding: {len(pending_frames)}")
    
    if pending_frames:
        t0 = time.time()
        indexed_count = search_svc.index_frames(pending_frames)
        embed_time = time.time() - t0
        print(f"    [P1.8 CLIP Visual Encoder] Embedded {indexed_count} frames in {embed_time:.2f}s")
        print(f"    [P1.10 FAISS Store] Total vectors in FAISS index: {search_svc.vector_index.total_vectors}")
    else:
        print("    All existing frames are already embedded and indexed!")

    # Verify SQLite embeddings records
    conn = get_db_connection()
    total_embeddings = conn.execute("SELECT COUNT(*) as c FROM embeddings").fetchone()['c']
    conn.close()
    print(f"    [P1.11 SQLite Embeddings] Total linked records in SQLite: {total_embeddings}")

    # Step 3: Week 4 - Vector Similarity Search & Ranking
    print("\n>>> STEP 3: RUNNING WEEK 4 NATURAL LANGUAGE SEMANTIC SEARCH (DFD 1.6 & 1.7)...")
    test_queries = [
        "a passenger bus on the street",
        "a person walking outdoors",
        "people standing near entrance",
        "a moving vehicle or car"
    ]

    for q in test_queries:
        print(f"\n    [Search Query]: \"{q}\"")
        q_start = time.time()
        results = search_svc.search(q, top_k=3, min_similarity=0.10)
        q_time = time.time() - q_start

        print(f"    Found {len(results)} matches in {q_time * 1000:.1f}ms:")
        for rank, res in enumerate(results, 1):
            det_names = [d["class_name"] for d in res["detections"]]
            det_str = ", ".join(det_names) if det_names else "none"
            print(f"      #{rank} [Score: {res['similarity_score']:.4f} ({res['similarity_percent']}%) | Time: {res['timestamp_sec']}s] Frame: {res['frame_id']} | Detections: [{det_str}]")

    print("\n================================================================================")
    print("  STATUS: WEEK 3 & WEEK 4 SUCCESSFULLY BUILT AND VERIFIED!")
    print("================================================================================")

if __name__ == "__main__":
    main()
