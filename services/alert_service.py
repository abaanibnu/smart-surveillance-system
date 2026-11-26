import cv2
import uuid
import math
import numpy as np
from datetime import datetime
from pathlib import Path

from database.db import (
    insert_alert,
    get_all_alerts,
    get_alert_by_id,
    update_alert_status,
    get_alerts_summary,
    get_frames_by_video,
    get_detections_by_frame
)

class RuleBasedAlertEngine:
    """
    Week 8: Rule-Based Security Engine & Real-Time Alert System (Figure 5.13: AlertSystem).
    Evaluates spatial polygon geofencing, temporal schedules, luggage abandonment,
    loitering, and restricted zone breaches.
    """

    # Configurable Default Spatial Geofence Zones (Normalized [0.0 - 1.0] coordinates)
    DEFAULT_ZONES = {
        "cam_01_gate": [
            [0.0, 0.0],
            [0.65, 0.0],
            [0.65, 0.65],
            [0.0, 0.65]
        ],
        "cam_02_corridor": [
            [0.15, 0.15],
            [0.85, 0.15],
            [0.85, 0.85],
            [0.15, 0.85]
        ]
    }

    # Configurable Security Rules Metadata
    RULE_DEFINITIONS = {
        "RULE_RESTRICTED_ZONE": {
            "name": "Restricted Geofence Intrusion",
            "description": "Triggered when a person breaches a marked security polygon.",
            "severity": "CRITICAL",
            "target_classes": ["person"]
        },
        "RULE_AFTER_HOURS": {
            "name": "Unauthorized After-Hours Presence",
            "description": "Triggered when individuals are detected outside 08:00 - 18:00 operating hours.",
            "severity": "HIGH",
            "target_classes": ["person"]
        },
        "RULE_UNATTENDED_OBJECT": {
            "name": "Unattended Luggage / Abandoned Package",
            "description": "Triggered when a bag/suitcase is stationary with no owner within proximity radius.",
            "severity": "HIGH",
            "target_classes": ["backpack", "handbag", "suitcase"]
        },
        "RULE_LOITERING": {
            "name": "Suspicious Loitering Detection",
            "description": "Triggered when a person lingers in a surveillance sector for > 5.0 seconds.",
            "severity": "MEDIUM",
            "target_classes": ["person"]
        },
        "RULE_VEHICLE_IN_PEDESTRIAN_ZONE": {
            "name": "Vehicle in Pedestrian Zone",
            "description": "Triggered when a car, motorcycle, or bus enters a pedestrian-only corridor.",
            "severity": "HIGH",
            "target_classes": ["car", "bus", "truck", "motorcycle"]
        }
    }

    def __init__(self, operating_hours=(8, 18)):
        self.operating_hours = operating_hours  # (start_hour, end_hour)
        self.zones = dict(self.DEFAULT_ZONES)

    def is_inside_polygon(self, point: tuple, polygon_pts: list, frame_w: int = 1920, frame_h: int = 1080) -> bool:
        """
        Checks if a point (x, y) falls inside the polygon geofence using OpenCV pointPolygonTest.
        Supports normalized [0, 1] polygon points.
        """
        scaled_polygon = []
        for pt in polygon_pts:
            px = int(pt[0] * frame_w) if pt[0] <= 1.0 else int(pt[0])
            py = int(pt[1] * frame_h) if pt[1] <= 1.0 else int(pt[1])
            scaled_polygon.append([px, py])

        poly_arr = np.array(scaled_polygon, dtype=np.int32)
        test_pt = (int(point[0]), int(point[1]))
        return cv2.pointPolygonTest(poly_arr, test_pt, False) >= 0

    def evaluate_frame(self, camera_id: str, frame_id: str, detections: list, timestamp_sec: float,
                       frame_w: int = 1920, frame_h: int = 1080,
                       simulated_after_hours: bool = False, is_demo: bool = False) -> list:
        """
        Evaluates all security rules on detections belonging to a single frame.
        Inserts any triggered alerts into SQLite and returns the list of generated alert records.
        """
        triggered_alerts = []
        persons = [d for d in detections if d["class_name"].lower() == "person"]
        luggage = [d for d in detections if d["class_name"].lower() in ["backpack", "handbag", "suitcase"]]
        vehicles = [d for d in detections if d["class_name"].lower() in ["car", "bus", "truck", "motorcycle", "bicycle"]]

        # -------------------------------------------------------------
        # Rule 1: Restricted Zone Intrusion (Polygon Geofence)
        # -------------------------------------------------------------
        zone_polygon = self.zones.get(camera_id)
        if zone_polygon and persons:
            for p in persons:
                # Use bottom-center point of bounding box
                bx = (p["bbox_x1"] + p["bbox_x2"]) / 2.0
                by = p["bbox_y2"]
                if self.is_inside_polygon((bx, by), zone_polygon, frame_w, frame_h):
                    alert_id = f"alt_{uuid.uuid4().hex[:10]}"
                    details = f"Person detected inside restricted geofence at ({int(bx)}, {int(by)}). Confidence: {int(p['confidence']*100)}%"
                    insert_alert(
                        alert_id=alert_id,
                        rule_id="RULE_RESTRICTED_ZONE",
                        alert_type="Restricted Geofence Intrusion",
                        camera_id=camera_id,
                        frame_id=frame_id,
                        details=details,
                        severity="CRITICAL",
                        status="active",
                        is_demo=1 if is_demo else 0
                    )
                    triggered_alerts.append({
                        "alert_id": alert_id,
                        "rule_id": "RULE_RESTRICTED_ZONE",
                        "severity": "CRITICAL",
                        "details": details
                    })

        # -------------------------------------------------------------
        # Rule 2: After-Hours Security Rule
        # -------------------------------------------------------------
        current_hour = datetime.now().hour
        is_after_hours = simulated_after_hours or (current_hour < self.operating_hours[0] or current_hour >= self.operating_hours[1])
        if is_after_hours and persons:
            for p in persons:
                alert_id = f"alt_{uuid.uuid4().hex[:10]}"
                details = f"Unauthorized person detected outside operating hours ({self.operating_hours[0]}:00 - {self.operating_hours[1]}:00). Conf: {int(p['confidence']*100)}%"
                insert_alert(
                    alert_id=alert_id,
                    rule_id="RULE_AFTER_HOURS",
                    alert_type="Unauthorized After-Hours Presence",
                    camera_id=camera_id,
                    frame_id=frame_id,
                    details=details,
                    severity="HIGH",
                    status="active",
                    is_demo=1 if is_demo else 0
                )
                triggered_alerts.append({
                    "alert_id": alert_id,
                    "rule_id": "RULE_AFTER_HOURS",
                    "severity": "HIGH",
                    "details": details
                })

        # -------------------------------------------------------------
        # Rule 3: Unattended Object / Luggage Abandonment
        # -------------------------------------------------------------
        proximity_threshold = 280.0  # Pixels distance
        if luggage:
            for item in luggage:
                item_cx = (item["bbox_x1"] + item["bbox_x2"]) / 2.0
                item_cy = (item["bbox_y1"] + item["bbox_y2"]) / 2.0

                min_dist = float("inf")
                for p in persons:
                    p_cx = (p["bbox_x1"] + p["bbox_x2"]) / 2.0
                    p_cy = (p["bbox_y1"] + p["bbox_y2"]) / 2.0
                    d = math.hypot(item_cx - p_cx, item_cy - p_cy)
                    if d < min_dist:
                        min_dist = d

                # If no persons present or nearest person is too far away
                if not persons or min_dist > proximity_threshold:
                    alert_id = f"alt_{uuid.uuid4().hex[:10]}"
                    dist_text = f"{int(min_dist)}px" if persons else "no person in frame"
                    details = f"Unattended {item['class_name']} detected without owner in close proximity (Distance: {dist_text})"
                    insert_alert(
                        alert_id=alert_id,
                        rule_id="RULE_UNATTENDED_OBJECT",
                        alert_type="Unattended Luggage / Abandoned Package",
                        camera_id=camera_id,
                        frame_id=frame_id,
                        details=details,
                        severity="HIGH",
                        status="active",
                        is_demo=1 if is_demo else 0
                    )
                    triggered_alerts.append({
                        "alert_id": alert_id,
                        "rule_id": "RULE_UNATTENDED_OBJECT",
                        "severity": "HIGH",
                        "details": details
                    })

        # -------------------------------------------------------------
        # Rule 4: Vehicle in Pedestrian Zone
        # -------------------------------------------------------------
        if "corridor" in camera_id.lower() and vehicles:
            for v in vehicles:
                alert_id = f"alt_{uuid.uuid4().hex[:10]}"
                details = f"Vehicle ({v['class_name']}) detected in restricted pedestrian corridor {camera_id}. Conf: {int(v['confidence']*100)}%"
                insert_alert(
                    alert_id=alert_id,
                    rule_id="RULE_VEHICLE_IN_PEDESTRIAN_ZONE",
                    alert_type="Vehicle in Pedestrian Zone",
                    camera_id=camera_id,
                    frame_id=frame_id,
                    details=details,
                    severity="HIGH",
                    status="active",
                    is_demo=1 if is_demo else 0
                )
                triggered_alerts.append({
                    "alert_id": alert_id,
                    "rule_id": "RULE_VEHICLE_IN_PEDESTRIAN_ZONE",
                    "severity": "HIGH",
                    "details": details
                })

        return triggered_alerts

    def evaluate_video(self, video_id: str, simulated_after_hours: bool = False) -> dict:
        """
        Runs comprehensive rule evaluation across all sampled frames of an ingested video.
        """
        frames = get_frames_by_video(video_id)
        if not frames:
            return {"success": False, "error": f"No frames found for video {video_id}", "alerts_count": 0}

        total_alerts = 0
        camera_id = frames[0].get("camera_id", "cam_01")
        
        # Track person dwell time for loitering check
        person_presence_count = 0
        first_person_frame = None

        for idx, f in enumerate(frames):
            frame_id = f["frame_id"]
            dets = get_detections_by_frame(frame_id)
            timestamp = f["timestamp_sec"]

            # Evaluate standard spatial and temporal rules
            frame_alerts = self.evaluate_frame(
                camera_id=camera_id,
                frame_id=frame_id,
                detections=dets,
                timestamp_sec=timestamp,
                simulated_after_hours=simulated_after_hours,
                is_demo=False
            )
            total_alerts += len(frame_alerts)

            # Loitering Rule check
            persons = [d for d in dets if d["class_name"].lower() == "person"]
            if persons:
                person_presence_count += 1
                if first_person_frame is None:
                    first_person_frame = f
                
                # If person persists across >= 4 consecutive frames (e.g. >= 4.0 seconds)
                if person_presence_count >= 4:
                    alert_id = f"alt_{uuid.uuid4().hex[:10]}"
                    details = f"Person detected lingering continuously for {person_presence_count * 1.0:.1f}s in surveillance zone {camera_id}"
                    insert_alert(
                        alert_id=alert_id,
                        rule_id="RULE_LOITERING",
                        alert_type="Suspicious Loitering Detection",
                        camera_id=camera_id,
                        frame_id=frame_id,
                        details=details,
                        severity="MEDIUM",
                        status="active",
                        is_demo=0
                    )
                    total_alerts += 1
                    person_presence_count = 0  # Reset counter after triggering
            else:
                person_presence_count = 0
                first_person_frame = None

        return {
            "success": True,
            "video_id": video_id,
            "evaluated_frames": len(frames),
            "alerts_triggered": total_alerts
        }

    def simulate_demo_alert(self, scenario: str = "zone_intrusion", camera_id: str = "cam_01_gate") -> dict:
        """
        Creates a realistic simulated security alert with is_demo=1 for testing & viva presentation.
        Scenarios: 'zone_intrusion', 'after_hours', 'unattended_bag', 'loitering', 'vehicle_breach'
        """
        alert_id = f"demo_alt_{uuid.uuid4().hex[:8]}"

        scenarios_map = {
            "zone_intrusion": {
                "rule_id": "RULE_RESTRICTED_ZONE",
                "alert_type": "Restricted Geofence Intrusion",
                "severity": "CRITICAL",
                "details": "DEMO SCENARIO: Individual crossed marked campus security perimeter fence at entrance gate."
            },
            "after_hours": {
                "rule_id": "RULE_AFTER_HOURS",
                "alert_type": "Unauthorized After-Hours Presence",
                "severity": "HIGH",
                "details": "DEMO SCENARIO: Motion detected inside computer lab corridor at 23:45 outside operating hours."
            },
            "unattended_bag": {
                "rule_id": "RULE_UNATTENDED_OBJECT",
                "alert_type": "Unattended Luggage / Abandoned Package",
                "severity": "HIGH",
                "details": "DEMO SCENARIO: Unattended backpack detected near building entrance without owner for > 30s."
            },
            "loitering": {
                "rule_id": "RULE_LOITERING",
                "alert_type": "Suspicious Loitering Detection",
                "severity": "MEDIUM",
                "details": "DEMO SCENARIO: Subject observed hovering stationary near server room corridor for > 15s."
            },
            "vehicle_breach": {
                "rule_id": "RULE_VEHICLE_IN_PEDESTRIAN_ZONE",
                "alert_type": "Vehicle in Pedestrian Zone",
                "severity": "HIGH",
                "details": "DEMO SCENARIO: Unauthorized motorcycle detected navigating campus pedestrian walkway."
            }
        }

        spec = scenarios_map.get(scenario, scenarios_map["zone_intrusion"])

        insert_alert(
            alert_id=alert_id,
            rule_id=spec["rule_id"],
            alert_type=spec["alert_type"],
            camera_id=camera_id,
            frame_id=None,
            details=spec["details"],
            severity=spec["severity"],
            status="active",
            is_demo=1
        )

        return {
            "success": True,
            "alert_id": alert_id,
            "scenario": scenario,
            **spec
        }

# Global singleton
_alert_engine_instance = None

def get_alert_engine() -> RuleBasedAlertEngine:
    global _alert_engine_instance
    if _alert_engine_instance is None:
        _alert_engine_instance = RuleBasedAlertEngine()
    return _alert_engine_instance
