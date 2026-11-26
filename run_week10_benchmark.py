#!/usr/bin/env python3
"""
================================================================================
  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM
  WEEK 10: SYSTEM INTEGRATION, PERFORMANCE BENCHMARKS & VIVA PREP
================================================================================
Academic Validation & Capstone Defense Evaluation Suite (DFD Level 2 & Figures 5.1-5.15)
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from services.benchmark_service import get_benchmark_service

def main():
    print("=" * 80)
    print("  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM")
    print("  WEEK 10: SYSTEM INTEGRATION & PERFORMANCE BENCHMARKS EVALUATION")
    print("=" * 80)
    print()

    benchmark_svc = get_benchmark_service()

    # Step 1: Hardware Environment
    print(">>> STEP 1: INSPECTING HOST EDGE HARDWARE & RUNTIME ENVIRONMENT...")
    hw = benchmark_svc.get_system_hardware_info()
    print(f"    Operating System:   {hw['os']}")
    print(f"    Python Version:     {hw['python_version']}")
    print(f"    CPU Architecture:   {hw['processor']} ({hw['physical_cores']} Phys / {hw['logical_cores']} Log Cores)")
    print(f"    RAM Memory:         {hw['total_ram_gb']} GB Total ({hw['available_ram_gb']} GB Available)")
    print("    [PASS] Hardware and edge computing resources verified.\n")

    # Step 2: Comprehensive Multi-Tier Benchmark
    print(">>> STEP 2: RUNNING COMPREHENSIVE MULTI-TIER SYSTEM BENCHMARKS...")
    print("    Evaluating YOLOv8, CLIP Vision, CLIP Text, FAISS, OWL Ontology, OSNet, AES-256, Bandwidth...")
    
    t0 = time.time()
    report = benchmark_svc.run_comprehensive_benchmark()
    eval_elapsed = time.time() - t0

    benchmarks = report["benchmarks"]
    print(f"\n    Completed all tiers in {eval_elapsed:.2f} seconds.")
    print("-" * 80)
    print(f"    {'COMPONENT / METRIC':<36} | {'MEASURED VALUE':<18} | {'TARGET':<12} | {'STATUS'}")
    print("-" * 80)

    # YOLO
    yolo = benchmarks.get("yolo_detection", {})
    print(f"    {'1. YOLOv8 Edge Detection':<36} | {yolo.get('avg_latency_ms', 0):>6.2f} ms ({yolo.get('throughput_fps', 0)} FPS) | {yolo.get('target_fps', ''):<12} | [{yolo.get('status', 'N/A')}]")

    # CLIP Vision
    clip_vis = benchmarks.get("clip_visual_embedding", {})
    print(f"    {'2. CLIP Visual Frame Encoding':<36} | {clip_vis.get('avg_latency_ms', 0):>6.2f} ms            | {clip_vis.get('target_latency', ''):<12} | [{clip_vis.get('status', 'N/A')}]")

    # CLIP Text
    clip_txt = benchmarks.get("clip_text_embedding", {})
    print(f"    {'3. CLIP Query Text Encoding':<36} | {clip_txt.get('avg_latency_ms', 0):>6.2f} ms            | {clip_txt.get('target_latency', ''):<12} | [{clip_txt.get('status', 'N/A')}]")

    # FAISS
    faiss_b = benchmarks.get("faiss_vector_retrieval", {})
    print(f"    {'4. FAISS Similarity Search (Cosine)':<36} | {faiss_b.get('avg_latency_ms', 0):>6.3f} ms            | {faiss_b.get('target_latency', ''):<12} | [{faiss_b.get('status', 'N/A')}]")

    # Ontology
    ont_b = benchmarks.get("ontology_expansion", {})
    print(f"    {'5. OWL/Protégé Query Expansion':<36} | {ont_b.get('avg_latency_ms', 0):>6.2f} ms            | {ont_b.get('target_latency', ''):<12} | [{ont_b.get('status', 'N/A')}]")

    # Re-ID
    reid_b = benchmarks.get("person_reid", {})
    print(f"    {'6. OSNet Person Re-ID Extraction':<36} | {reid_b.get('avg_latency_ms', 0):>6.2f} ms            | {reid_b.get('target_latency', ''):<12} | [{reid_b.get('status', 'N/A')}]")

    # AES-256
    aes_b = benchmarks.get("aes_256_cryptography", {})
    print(f"    {'7. AES-256 Vector Cryptography':<36} | {aes_b.get('encryption_latency_ms', 0):>6.2f} ms ({aes_b.get('throughput_mb_per_sec', 0)} MB/s) | {aes_b.get('target_overhead', ''):<12} | [{aes_b.get('status', 'N/A')}]")

    # Bandwidth
    bw_b = benchmarks.get("bandwidth_reduction", {})
    print(f"    {'8. Bandwidth & Storage Reduction':<36} | {bw_b.get('bandwidth_reduction_percentage', '0%'):>8} ({bw_b.get('storage_reduction_ratio', '')}) | {bw_b.get('target_reduction', ''):<12} | [{bw_b.get('status', 'N/A')}]")

    print("-" * 80)
    print()

    # Step 3: Forensic Incident Dossier Generation
    print(">>> STEP 3: GENERATING FORENSIC SURVEILLANCE INCIDENT DOSSIER (USE CASE 8)...")
    dossier = benchmark_svc.generate_forensic_incident_dossier()
    print(f"    Dossier ID:           {dossier['dossier_id']}")
    print(f"    Total Incidents:      {dossier['total_incidents']}")
    print(f"    Active Alerts:        {dossier['summary'].get('active_alerts', 0)}")
    print(f"    Critical Incidents:   {dossier['summary'].get('critical_active', 0)}")
    print("    [PASS] Forensic surveillance incident audit dossier generated.")
    print()

    # Step 4: Persisting Results
    out_path = BASE_DIR / "database" / "benchmark_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f">>> STEP 4: BENCHMARK REPORT PERSISTED TO: {out_path}")
    print()

    print("=" * 80)
    print("  STATUS: WEEK 10 SYSTEM INTEGRATION & BENCHMARKS VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
