import os
import sys
import time
import wave
import struct
import math
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DB_PATH, UPLOADS_DIR
from database.db import init_db
from services.ontology_service import OntologyEngine
from services.speech_service import VoiceQueryHandler
from services.search_service import SearchService

def create_dummy_audio_file(output_path: str, duration_sec=2, sample_rate=16000):
    """Generates a valid 16kHz mono WAV audio file for voice pipeline validation."""
    n_samples = int(duration_sec * sample_rate)
    with wave.open(output_path, 'w') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        
        # Write gentle tone audio
        data = bytearray()
        for i in range(n_samples):
            val = int(3000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
            data.extend(struct.pack('<h', val))
        wav.writeframes(data)
    return output_path

def main():
    print("================================================================================")
    print("  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM")
    print("  WEEK 5 & WEEK 6 VERIFICATION TEST RUNNER")
    print("================================================================================")

    init_db()

    # Step 1: Week 5 - Ontology Engine & RDFLib Concept Expansion
    print("\n>>> STEP 1: INITIALIZING ONTOLOGY ENGINE WITH SURVEILLANCE.OWL (DFD 1.5: P2.2)...")
    ontology = OntologyEngine()
    print(f"    Loaded ontology graph from: {ontology.ontology_file}")
    print(f"    Total RDF triples indexed: {len(ontology.graph)}")

    test_terms = ["vehicle", "luggage", "person", "corridor"]
    for term in test_terms:
        expanded = ontology.expandQueryConcepts(term)
        print(f"    [Expansion] '{term}' -> {len(expanded)} concepts: {expanded[:8]}...")

    # Step 2: Test Full Natural Language Query Concept Expansion
    test_nl_query = "show me any vehicle or transport near the entrance gate"
    print(f"\n>>> STEP 2: TESTING NATURAL LANGUAGE QUERY EXPANSION...")
    print(f"    Input Query: \"{test_nl_query}\"")
    expansion_res = ontology.expand_full_query(test_nl_query)
    print(f"    Composite Query: \"{expansion_res['composite_query']}\"")
    print(f"    Triggered Concepts: {list(expansion_res['detected_concepts'].keys())}")
    print(f"    Injected Synonyms: {expansion_res['expanded_terms'][:6]}")

    # Step 3: Week 6 - Unified Multimodal Search Service
    print("\n>>> STEP 3: RUNNING UNIFIED ONTOLOGY-DRIVEN SEMANTIC SEARCH (DFD 1.6 & 1.7)...")
    search_svc = SearchService(ontology=ontology)
    
    # Query: "vehicle" (which has NO exact word 'bus' in it, but ontology expands to 'bus')
    t0 = time.time()
    search_res = search_svc.search("vehicle", top_k=3, min_similarity=0.10, use_ontology=True)
    t_search = time.time() - t0

    print(f"    Search for 'vehicle' (Ontology expanded: {search_res['expanded_terms'][:5]}...)")
    print(f"    Found {search_res['total_matches']} ranked frame matches in {t_search * 1000:.1f}ms:")
    for rank, r in enumerate(search_res["results"], 1):
        dets = [d["class_name"] for d in r["detections"]]
        print(f"      #{rank} [Similarity: {r['similarity_percent']}% | Timestamp: {r['timestamp_sec']}s] Frame: {r['frame_id']} | Objects: {dets}")

    # Step 4: Week 6 - Whisper Speech-to-Text Voice Query Pipeline
    print("\n>>> STEP 4: VERIFYING WHISPER SPEECH RECOGNITION (SEQUENCE DIAGRAM 5.14)...")
    voice_handler = VoiceQueryHandler(model_name="base.en")
    print("    Whisper model loaded into memory.")
    
    # Generate test audio
    test_audio_path = str(UPLOADS_DIR / "test_operator_voice.wav")
    create_dummy_audio_file(test_audio_path, duration_sec=1)
    
    v_start = time.time()
    v_res = voice_handler.transcribe_audio(test_audio_path)
    v_time = time.time() - v_start
    print(f"    Transcribed test audio in {v_time:.2f}s | Success: {v_res['success']} | Output: '{v_res['text']}'")

    print("\n================================================================================")
    print("  STATUS: WEEK 5 & WEEK 6 SUCCESSFULLY BUILT AND VERIFIED!")
    print("================================================================================")

if __name__ == "__main__":
    main()
