# Capstone Project Viva Preparation & Defense Guide

**Project Title**: *Ontology-Driven AI Semantic Video Search and Smart Surveillance System*  
**Degree**: Bachelor of Technology in Computer Science and Engineering  
**Academic Milestone**: Week 10 Final Capstone Integration & Academic Evaluation  

---

## 1. Executive Summary & Problem-Solution Fit

### The Problem in Traditional Video Surveillance
- **Manual Scrubbing**: Finding specific incidents in multi-hour CCTV archives requires operators to manually watch footage at $2\times$ or $4\times$ speed, which is slow, error-prone, and labor-intensive.
- **Bandwidth & Storage Bottlenecks**: Continuously streaming raw 1080p surveillance video ($25\text{ FPS} \approx 4\text{ Mbps}$) over networks to cloud servers consumes $\approx 1.8\text{ GB/hour}$ per camera, driving up operational expenses.
- **Keyword Blindness**: Conventional database searches only filter by exact camera IDs or time windows, lacking natural language semantic understanding (e.g. searching *"person in blue coat holding laptop"*).

### Our Edge AI Solution
- **Local Edge Processing**: Video feeds are sampled ($1\text{ FPS}$ intervals) and processed locally on the edge device using lightweight **YOLOv8** object detection and **CLIP (ViT-B/32)** visual embedding.
- **Ontology Concept Expansion (Protégé & OWL)**: Queries are expanded taxonomically before vector search (e.g., query *"vehicle"* automatically expands to *car, bus, motorcycle, bicycle, truck*).
- **Sub-Millisecond Vector Retrieval**: **FAISS** indexes normalized 512-dimensional CLIP embeddings for inner-product cosine similarity search in $< 1\text{ ms}$.
- **Multimodal Voice Interaction**: Integrated **OpenAI Whisper** transcribes dictated speech queries into text.
- **Cross-Camera Person Re-ID**: **OSNet** deep metric embeddings track identical individuals across multiple camera views without facial recognition.
- **Rule-Based Security Engine**: Spatial polygon geofences, temporal schedules (after-hours), and loitering/abandonment detection trigger automated alerts.
- **Edge Security Shield**: Authenticated **AES-256** vector encryption at rest guarantees zero plaintext vector leakage to disk.
- **Bandwidth Optimization**: Transmitting only frame metadata, vectors, and thumbnails saves **$91.3\%$ bandwidth** ($149.4\text{ MB/hr}$ vs $1,716.6\text{ MB/hr}$ raw stream).

---

## 2. Mathematical Formulations for Viva Defense

### A. Vector Similarity Formulation (CLIP & FAISS)
Given a 512-dimensional query vector $\vec{q} \in \mathbb{R}^{512}$ and stored frame embedding $\vec{f} \in \mathbb{R}^{512}$, both normalized such that $\|\vec{q}\|_2 = \|\vec{f}\|_2 = 1.0$, the cosine similarity $S_C(\vec{q}, \vec{f})$ is calculated via the inner dot product:

$$S_C(\vec{q}, \vec{f}) = \frac{\vec{q} \cdot \vec{f}}{\|\vec{q}\|_2 \|\vec{f}\|_2} = \sum_{i=1}^{512} q_i \cdot f_i$$

### B. Bandwidth Optimization Equation
Let $B_{\text{raw}}$ be the data rate of continuous 1080p H.264 video streaming at $25\text{ FPS}$ ($4.0\text{ Mbps} \approx 1,716.6\text{ MB/hr}$), and $B_{\text{edge}}$ be the total edge payload ($1\text{ FPS}$ JPEG thumbnail $\approx 40\text{ KB}$ + 512-dim FP32 vector $\approx 2.05\text{ KB}$ + SQLite record $\approx 0.5\text{ KB} = 42.55\text{ KB/s} \approx 149.4\text{ MB/hr}$). The bandwidth reduction percentage $\beta$ is:

$$\beta = \left( \frac{B_{\text{raw}} - B_{\text{edge}}}{B_{\text{raw}}} \right) \times 100\% = \left( \frac{1716.6 - 149.4}{1716.6} \right) \times 100\% = 91.3\%$$

### C. Authenticated AES-256 Encryption at Rest
Vector storage is protected using Fernet authenticated symmetric encryption:

$$\text{Ciphertext} = \text{IV} \parallel \text{AES}_{256\text{-CBC}}(K_{\text{enc}}, P) \parallel \text{HMAC-SHA256}(K_{\text{mac}}, \text{IV} \parallel \text{AES}_{256\text{-CBC}}(K_{\text{enc}}, P))$$

Where $K_{\text{enc}}$ and $K_{\text{mac}}$ are derived from the 256-bit Edge Master Key stored at `database/edge_master.key`.

---

## 3. Architecture & Diagrams Alignment

| Diagram | Figure | Implementation in Codebase |
|---|---|---|
| **Level 0 DFD** | Fig 5.2 | High-level data exchange between Operator, Edge Device, SQLite, and FAISS. |
| **Level 1 DFD** | Fig 5.3 | P1 (Ingestion & Edge AI), P2 (Search & Display), D1 (FAISS), D2 (SQLite). |
| **Video Ingestion** | Fig 5.4, 5.5 | `services/video_service.py` (P1.1 - P1.6): OpenCV sampling, format verification. |
| **Detection & Embedding** | Fig 5.6, 5.7 | `models/detector.py` (YOLOv8) & `models/embedder.py` (CLIP image encoder). |
| **Query & Expansion** | Fig 5.8, 5.9 | `services/ontology_service.py` (Protégé OWL expansion) & `models/faiss_index.py`. |
| **Use Case Diagram** | Fig 5.12 | 8 Use Cases: Ingest, Detect, Semantic Search, Voice Search, Re-ID, Alerts, Forensic Player, Audit Dossier. |
| **Class Diagram** | Fig 5.13 | `Video`, `Frame`, `Detection`, `CLIPEmbedder`, `FAISSIndex`, `OntologyEngine`, `PersonReID`, `AlertSystem`. |
| **Sequence Diagram** | Fig 5.14 | Voice/Text Query: Operator $\rightarrow$ Whisper $\rightarrow$ Ontology $\rightarrow$ CLIP $\rightarrow$ FAISS $\rightarrow$ SQLite $\rightarrow$ UI. |
| **Activity Diagram** | Fig 5.15 | Dual-path workflow: Ingestion Path vs Search/Review Path. |

---

## 4. Top 12 Viva Questions & High-Scoring Answers

#### Q1: Why use an Ontology instead of searching directly with CLIP?
> **Answer**: While CLIP has strong zero-shot vision-language alignment, natural surveillance queries frequently use hypernyms (e.g., searching *"vehicle"* when the frame contains a *"bus"* or *"motorcycle"*). Our Protégé OWL surveillance ontology defines domain taxonomies and transitive relationships. Expanding the query with domain-specific subclasses (e.g. `Vehicle` $\rightarrow$ `Car`, `Bus`, `Motorcycle`) before visual embedding dramatically increases semantic recall (Recall@5 improved by $> 25\%$) without losing precision.

#### Q2: How does the system achieve 91.3% bandwidth reduction?
> **Answer**: Standard CCTV cameras continuously stream every frame over the network ($25\text{ FPS}$ at $1080\text{p} \approx 1,716\text{ MB/hr}$). Our Edge AI architecture shifts model inference to the edge camera node. We sample frames at $1\text{ FPS}$, extract 512-dimensional vector representations locally, and transmit only the compact vector, frame thumbnail, and structured metadata ($149.4\text{ MB/hr}$). The raw video stays stored locally, achieving a $91.3\%$ ($11.5\times$) bandwidth reduction.

#### Q3: Why is FAISS preferred over traditional SQL databases for video retrieval?
> **Answer**: Relational databases (SQLite, MySQL) excel at scalar filtering (e.g., `WHERE timestamp > 10.0`), but cannot perform high-dimensional similarity searches. Visual embeddings are continuous 512-dimensional vectors. FAISS (Facebook AI Similarity Search) implements optimized C++ SIMD vector indexing, allowing inner-product cosine similarity searches over hundreds of candidate vectors in under $0.5\text{ milliseconds}$.

#### Q4: How is data protected at rest?
> **Answer**: In Week 9, we integrated standard AES-256 (Fernet CBC mode with HMAC-SHA256 authenticated integrity). Vector embeddings on disk are serialized in-memory and encrypted as `surveillance_vectors.enc`. Even if the storage media is stolen or compromised, no raw visual feature embeddings can be extracted without the 256-bit Master Key.

#### Q5: How does the Person Re-Identification system track across cameras without facial recognition?
> **Answer**: Surveillance cameras are typically mounted at high angles where facial features are heavily obscured or low resolution. Our Re-ID pipeline uses OSNet deep metric visual backbones to extract 512-dimensional whole-body feature descriptors (clothing color, torso proportions, carry-on accessories) from detected pedestrian bounding boxes, matching individuals across Camera 1 and Camera 2 via cosine distance matching.

#### Q6: What happens if an operator dictates a voice query in a noisy environment?
> **Answer**: We use OpenAI Whisper (`base.en`), which is trained on 680,000 hours of multilingual audio. Whisper utilizes an encoder-decoder Transformer that demonstrates high robustness to background noise and accents compared to conventional acoustic models.

#### Q7: What are the primary spatial and temporal security rules in the Alert Engine?
> **Answer**:
> 1. `RULE_RESTRICTED_ZONE`: Spatial polygon geofence checking if a detected person's centroid breaches restricted campus zones.
> 2. `RULE_AFTER_HOURS`: Temporal check verifying if human presence occurs outside permitted working hours.
> 3. `RULE_LUGGAGE_ABANDONMENT`: Measures spatial Euclidean distance between luggage (`backpack`, `suitcase`) and the nearest person centroid; if separation $> 0.25$ normalized distance persists, an alert is triggered.
> 4. `RULE_LOITERING`: Flags persons remaining within a localized bounding threshold across multiple consecutive frames.

#### Q8: How does the Forensic Video Review Player work?
> **Answer**: Figure 5.12 Use Case 5 allows interactive timestamp inspection. When an alert or search result is clicked, the HTML5 surveillance player fetches the clip from `GET /api/video/<video_id>/stream` and sets `video.currentTime = targetTimestamp`. Controls allow forensic frame-stepping ($-5\text{s}$, $-1\text{s}$, $+1\text{s}$, $+5\text{s}$) and target re-sync.

#### Q9: What are the measured latencies across the pipeline?
> **Answer**:
> - YOLOv8n Detection: $\approx 67.5\text{ ms}$ ($14.8\text{ FPS}$)
> - CLIP Visual Encoding: $\approx 124.7\text{ ms}$
> - CLIP Text Query Encoding: $\approx 91.9\text{ ms}$
> - FAISS Vector Similarity Search: $\approx 0.51\text{ ms}$
> - AES-256 Encryption Overhead: $\approx 3.5\text{ ms}$ ($71.1\text{ MB/s}$ throughput)

#### Q10: How do you prevent replay or tampering of forensic evidence?
> **Answer**: In Week 10, the system generates a tamper-evident Forensic Incident Dossier (`/api/benchmark/dossier`). Every alert includes the camera ID, frame ID, timestamp, and the SHA-256 cryptographic hash of the evidence vector storage, providing a legally verifiable chain of custody.

#### Q11: What were the hardware limitations encountered and how were they mitigated?
> **Answer**: We designed the pipeline to operate on consumer-grade quad-core CPUs without requiring a dedicated high-end GPU. By selecting YOLOv8n (nano), CLIP ViT-B/32, sampling at $1.0\text{s}$ intervals, and caching FAISS vector indices in memory, CPU inference remains responsive in real-time.

#### Q12: What future enhancements can be added to this project?
> **Answer**: Future directions include deploying over multiple distributed Raspberry Pi / Jetson Orin edge nodes with MQTT/gRPC synchronization, fine-tuning CLIP on specialized CCTV security datasets, and extending Re-ID tracking across pan-tilt-zoom (PTZ) optical zoom cameras.
