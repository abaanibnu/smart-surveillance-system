import os
import uuid
import cv2
from pathlib import Path
from ultralytics import YOLO
from config import YOLO_MODEL_NAME, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, TARGET_SURVEILLANCE_CLASSES, ANNOTATED_DIR
from database.db import insert_detections_batch

class EdgeDetector:
    """
    Implements DFD Level 1.3:
    - P1.7: YOLO Detection (people and common surveillance objects)
    - P1.9: Link Frame Data (associate detections with frame IDs in SQLite)
    """

    def __init__(self, model_weight=YOLO_MODEL_NAME, conf_thresh=CONFIDENCE_THRESHOLD, iou_thresh=IOU_THRESHOLD):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = YOLO(model_weight)

    def detect_in_frame(self, frame_path: str):
        """
        Runs YOLOv8 detection on a single frame file.
        Returns: list of dicts with detected target classes and coordinates.
        """
        results = self.model.predict(
            source=frame_path,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            save=False,
            verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                # Only retain targets defined in surveillance specification
                if cls_id in TARGET_SURVEILLANCE_CLASSES:
                    class_name = TARGET_SURVEILLANCE_CLASSES[cls_id]
                    conf = float(box.conf[0].item())
                    coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    detections.append({
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(conf, 4),
                        "bbox": [round(c, 2) for c in coords]
                    })

        return detections

    def annotate_frame(self, frame_path: str, detections: list, output_path: str = None) -> str:
        """
        Draws bounding boxes and labels on the image for debugging and operator review.
        """
        image = cv2.imread(frame_path)
        if image is None:
            raise ValueError(f"Cannot read frame at: {frame_path}")

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class_name"]
            conf = det["confidence"]
            label = f"{cls_name} {conf:.2f}"

            # Distinct colors: Person = Green, Vehicle = Blue, Bag/Luggage = Orange, Electronics = Red
            if cls_name == "person":
                color = (0, 255, 0)
            elif cls_name in ["car", "motorcycle", "bicycle", "bus", "truck"]:
                color = (255, 165, 0) # BGR
            elif cls_name in ["backpack", "handbag", "suitcase"]:
                color = (0, 140, 255)
            else:
                color = (0, 0, 255)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Label background & text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, max(0, y1 - 20)), (x1 + w + 4, max(20, y1)), color, -1)
            cv2.putText(image, label, (x1 + 2, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if output_path is None:
            frame_name = Path(frame_path).name
            output_path = str(ANNOTATED_DIR / f"annotated_{frame_name}")

        cv2.imwrite(output_path, image)
        return output_path

    def process_and_link_frames(self, frames_list: list, generate_annotated=True):
        """
        P1.9: Links detections with frame IDs and persists to SQLite.
        frames_list: list of dicts with keys 'frame_id' and 'storage_path'.
        """
        total_detections_count = 0
        all_detections_db = []
        results_summary = []

        for f in frames_list:
            frame_id = f["frame_id"]
            frame_path = f["storage_path"]

            detections = self.detect_in_frame(frame_path)
            annotated_path = None

            if generate_annotated and len(detections) > 0:
                annotated_path = self.annotate_frame(frame_path, detections)

            for d in detections:
                det_id = f"det_{uuid.uuid4().hex[:12]}"
                bbox = d["bbox"]
                all_detections_db.append((
                    det_id,
                    frame_id,
                    d["class_name"],
                    d["confidence"],
                    bbox[0], bbox[1], bbox[2], bbox[3]
                ))

            total_detections_count += len(detections)
            results_summary.append({
                "frame_id": frame_id,
                "frame_path": frame_path,
                "annotated_path": annotated_path,
                "detections": detections
            })

        if all_detections_db:
            insert_detections_batch(all_detections_db)

        return {
            "total_frames_analyzed": len(frames_list),
            "total_detections_linked": total_detections_count,
            "summary": results_summary
        }
