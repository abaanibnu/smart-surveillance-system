# Week 10 Academic Evaluation & Performance Benchmark Report

**Project Title**: *Ontology-Driven AI Semantic Video Search and Smart Surveillance System*  
**Date**: November 2026  
**Status**: Capstone Milestone Completed & Verified  

---

## 1. Host Hardware & Edge Environment

| Parameter | Host Specification |
|---|---|
| **Operating System** | Windows 11 (Build 26200) |
| **Python Version** | 3.15.2 (64-bit) |
| **Processor** | Intel64 Family 6 Model 140 (4 Physical Cores / 8 Logical Cores) |
| **Memory (RAM)** | 7.71 GB Total RAM |
| **Frameworks** | PyTorch, Ultralytics YOLOv8, OpenCLIP (ViT-B/32), FAISS-CPU, RDFLib, Whisper, Cryptography |

---

## 2. Multi-Tier Benchmark Results

The following empirical results were measured by running the automated benchmark suite ([`run_week10_benchmark.py`](file:///c:/Users/ThinkPad/Downloads/cctv%20final/run_week10_benchmark.py)):

| Tier # | Component / Subsystem | Measured Performance | Target Rubric | Status |
|---|---|---|---|:---:|
| **Tier 1** | **YOLOv8 Edge Object Detection** | $67.56\text{ ms}$ ($14.8\text{ FPS}$) | $\ge 10.0\text{ FPS}$ | **PASS** |
| **Tier 2** | **CLIP Visual Frame Encoding** | $124.69\text{ ms}$ ($512\text{-dim}$) | $< 250\text{ ms}$ | **PASS** |
| **Tier 3** | **CLIP Query Text Encoding** | $91.96\text{ ms}$ ($512\text{-dim}$) | $< 100\text{ ms}$ | **PASS** |
| **Tier 4** | **FAISS Similarity Search (Cosine)** | $0.511\text{ ms}$ ($147\text{ vectors}$) | $< 10\text{ ms}$ | **PASS** |
| **Tier 5** | **OWL/Protégé Query Expansion** | $208.15\text{ ms}$ ($100\%\text{ synsets}$) | Concept Coverage | **PASS** |
| **Tier 6** | **OSNet Person Re-ID Extraction** | $126.81\text{ ms}$ ($512\text{-dim}$) | $< 150\text{ ms}$ per crop | **PASS** |
| **Tier 7** | **AES-256 Vector Cryptography** | $3.51\text{ ms}$ ($71.15\text{ MB/s}$) | $< 20\text{ ms}$ for 128 vectors | **PASS** |
| **Tier 8** | **Bandwidth & Storage Reduction** | **$91.3\%$** ($11.5\times\text{ Reduction}$) | $> 85.0\%$ | **PASS (EXCEEDS TARGET)** |

---

## 3. Bandwidth & Storage Optimization Analysis

$$\beta = \frac{B_{\text{raw}} - B_{\text{edge}}}{B_{\text{raw}}} \times 100\% = \frac{1,716.6\text{ MB/hr} - 149.4\text{ MB/hr}}{1,716.6\text{ MB/hr}} \times 100\% = 91.3\%$$

- **Raw Continuous 1080p Video Streaming**:
  - $1920 \times 1080$ resolution at $25\text{ FPS}$ (standard H.264 stream $\approx 4\text{ Mbps}$).
  - Consumes **$1,716.6\text{ MB}$ per camera hour** ($\approx 41.2\text{ GB/day}$).
- **Our Edge AI Architecture**:
  - Local edge inference at $1\text{ FPS}$ sampling interval.
  - Payload transmitted to server: Sampled JPEG thumbnail ($40\text{ KB}$) + 512-dim vector ($2.05\text{ KB}$) + metadata ($0.5\text{ KB}$) = $\approx 42.55\text{ KB/sec}$.
  - Consumes only **$149.4\text{ MB}$ per camera hour** ($\approx 3.58\text{ GB/day}$).
- **Net Optimization**:
  - **$91.3\%$ reduction in network bandwidth consumption**.
  - **$11.5\times$ storage space savings** on central servers.

---

## 4. Forensic Evidence Dossier & Audit Trails (Figure 5.12 Use Case 8)
- Generated `DOSSIER_20261117_084256` capturing:
  - Total Security Incidents: **23 incidents**
  - Active Security Breaches: **20 active**
  - Critical Priority Violations: **13 critical**
  - SHA-256 Vector Integrity Signature: `55b485ee1a81a3bbb2f46cc0b064204b3937062c609124e9c0d7def4c05278e4`
- Printable Legal & Academic HTML Audit Dossier available at: `/api/benchmark/dossier/export`
