# Ontology-Driven AI Semantic Video Search & Smart Surveillance System
**APJ Abdul Kalam Technological University — Final Year B.Tech Project**

## Weeks 1, 2, 3 & 4 Implementation: Ingestion, Detection, Embedding & Semantic Search

This repository implements the end-to-end Edge AI video ingestion, YOLOv8 object detection, visual semantic embedding with CLIP, and similarity vector retrieval conforming directly to the Data Flow Diagrams (Level 1.1 to 1.7) and Class Diagram specifications.

---

### Implemented Subsystems

1. **Week 1: Ingestion, Validation & Sampling (DFD 1.1 & 1.2)**
   - `P1.1 - P1.3 Ingest`: Format validation, OpenCV frame-rate/resolution calculation, unique `video_id` creation.
   - `P1.4 - P1.6 Sampling`: Fixed time-interval sampling (1.0 sec), timestamp preservation, and registration in SQLite.

2. **Week 2: YOLOv8 Object Detection & Linking (DFD 1.3)**
   - `P1.7 Detection`: Ultralytics YOLOv8 nano detects surveillance objects: people, cars, buses, bikes, bags, laptops, phones.
   - `P1.9 Linking`: Associates detections with frame IDs in SQLite (`detections` table).
   - Visual Bounding Boxes: Automatically renders annotated frames to `static/annotated/`.

3. **Week 3: Visual Semantic Embedding (DFD 1.3 & 1.4)**
   - `P1.8 CLIP Visual Encoder`: OpenAI CLIP `ViT-B/32` encodes frames into normalized 512-dimensional vectors.
   - `P1.11 Metadata Storage`: Stores embedding records in SQLite (`embeddings` table) linking `frame_id` to vector index IDs.

4. **Week 4: Vector Indexing & Semantic Search (DFD 1.4, 1.6 & 1.7)**
   - `P1.10 FAISS Index`: High-dimensional vector index supporting cosine similarity via inner product.
   - `P2.3 CLIP Text Encoder`: Encodes natural language queries into 512-dim vectors.
   - `P2.4 - P2.6 Similarity Search & Ranking`: Matches query vectors against stored frame embeddings; selects Top-$K$ matches.
   - `P2.7 - P2.9 Result Display`: Maps matched vector IDs back to frame metadata, video timestamp, and YOLO detections.

---

### How to Run

1. **Start the Web Dashboard**:
   ```powershell
   python app.py
   ```
   Open your browser at **http://127.0.0.1:5000** to use the interactive dashboard.

2. **Run Verification Test**:
   ```powershell
   python run_week3_week4_test.py
   ```
