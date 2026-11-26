import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
FRAMES_DIR = STATIC_DIR / "frames"
ANNOTATED_DIR = STATIC_DIR / "annotated"
CROPS_DIR = STATIC_DIR / "person_crops"
DATABASE_DIR = BASE_DIR / "database"
DB_PATH = DATABASE_DIR / "surveillance.db"

# Create directories if they do not exist
for folder in [UPLOADS_DIR, FRAMES_DIR, ANNOTATED_DIR, CROPS_DIR, DATABASE_DIR]:
    os.makedirs(folder, exist_ok=True)

# Ingestion Settings
DEFAULT_SAMPLE_INTERVAL_SEC = 1.0  # Sample 1 frame per second
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Target Surveillance Classes (COCO IDs and labels)
TARGET_SURVEILLANCE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    63: "laptop",
    67: "cell phone"
}

# Detection Settings
YOLO_MODEL_NAME = "yolov8n.pt"  # Lightweight YOLOv8 nano model
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
