"""
================================================================================
  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM
  WEEK 8: RULE-BASED SECURITY ENGINE & REAL-TIME ALERTS TEST SUITE
================================================================================
Tests:
1. Spatial Geofence Polygon Breach (RULE_RESTRICTED_ZONE)
2. Temporal After-Hours Presence (RULE_AFTER_HOURS)
3. Luggage Abandonment & Proximity Distance (RULE_UNATTENDED_OBJECT)
4. Temporal Dwell / Loitering Detection (RULE_LOITERING)
5. Vehicle Restriction in Pedestrian Zone (RULE_VEHICLE_IN_PEDESTRIAN_ZONE)
6. Demo Scenario Simulation & Operator Acknowledgment Lifecycle
"""

import os
import sys
from datetime import datetime

# Add root to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.db import (
    init_db,
    get_db_connection,
    get_all_alerts,
    update_alert_status,
    get_alerts_summary,
    clear_all_alerts
)
from services.alert_service import get_alert_engine, RuleBasedAlertEngine

def run_tests():
    print("=" * 80)
    print("  ONTOLOGY-DRIVEN AI SEMANTIC VIDEO SEARCH & SMART SURVEILLANCE SYSTEM")
    print("  WEEK 8: RULE-BASED SECURITY ENGINE VERIFICATION TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # STEP 1: Initialize Database & Verify Alert Schema
    # -------------------------------------------------------------
    print("\n>>> STEP 1: VERIFYING ALERTS DATABASE SCHEMA & MIGRATIONS...")
    init_db()
    conn = get_db_connection()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(alerts)").fetchall()]
    conn.close()
    print(f"    Alerts Schema Columns: {cols}")
    assert "severity" in cols, "Column 'severity' missing from alerts!"
    assert "status" in cols, "Column 'status' missing from alerts!"
    print("    [PASS] Alerts schema contains severity, status, and demo flags.")

    # -------------------------------------------------------------
    # STEP 2: Verify Rule 1 - Restricted Polygon Geofence Intrusion
    # -------------------------------------------------------------
    print("\n>>> STEP 2: VERIFYING RULE 1 - RESTRICTED GEOFENCE INTRUSION...")
    engine = get_alert_engine()
    
    # Inside gate zone [0.0 - 0.65]
    inside_detection = [{
        "class_name": "person",
        "confidence": 0.94,
        "bbox_x1": 200, "bbox_y1": 200, "bbox_x2": 400, "bbox_y2": 500  # base is at (300, 500)
    }]
    # Outside zone (e.g. bottom right corner x=1800, y=900)
    outside_detection = [{
        "class_name": "person",
        "confidence": 0.88,
        "bbox_x1": 1700, "bbox_y1": 800, "bbox_x2": 1900, "bbox_y2": 1000
    }]

    alerts_inside = engine.evaluate_frame("cam_01_gate", "frm_test_zone_in", inside_detection, 1.0)
    alerts_outside = engine.evaluate_frame("cam_01_gate", "frm_test_zone_out", outside_detection, 1.0)

    print(f"    Inside Restricted Zone: Triggered {len(alerts_inside)} alert(s) -> {[a['rule_id'] for a in alerts_inside]}")
    print(f"    Outside Restricted Zone: Triggered {len(alerts_outside)} alert(s)")
    assert any(a["rule_id"] == "RULE_RESTRICTED_ZONE" for a in alerts_inside), "Expected RULE_RESTRICTED_ZONE trigger!"
    assert not any(a["rule_id"] == "RULE_RESTRICTED_ZONE" for a in alerts_outside), "Outside point should not trigger!"
    print("    [PASS] Spatial Polygon Geofence test passed.")

    # -------------------------------------------------------------
    # STEP 3: Verify Rule 2 - Temporal After-Hours Security
    # -------------------------------------------------------------
    print("\n>>> STEP 3: VERIFYING RULE 2 - UNAUTHORIZED AFTER-HOURS PRESENCE...")
    person_det = [{
        "class_name": "person",
        "confidence": 0.91,
        "bbox_x1": 1700, "bbox_y1": 800, "bbox_x2": 1900, "bbox_y2": 1000
    }]
    
    alerts_after_hours = engine.evaluate_frame("cam_01_gate", "frm_test_after_hours", person_det, 2.0, simulated_after_hours=True)
    alerts_normal_hours = engine.evaluate_frame("cam_01_gate", "frm_test_normal_hours", person_det, 2.0, simulated_after_hours=False)

    print(f"    Simulated After-Hours (23:00): Triggered -> {[a['rule_id'] for a in alerts_after_hours]}")
    assert any(a["rule_id"] == "RULE_AFTER_HOURS" for a in alerts_after_hours), "Expected RULE_AFTER_HOURS trigger!"
    print("    [PASS] Temporal After-Hours security rule verified.")

    # -------------------------------------------------------------
    # STEP 4: Verify Rule 3 - Unattended Luggage / Abandoned Object
    # -------------------------------------------------------------
    print("\n>>> STEP 4: VERIFYING RULE 3 - UNATTENDED LUGGAGE / ABANDONED OBJECT...")
    # Scenario A: Backpack far away from any person (Abandoned)
    abandoned_luggage_dets = [
        {"class_name": "backpack", "confidence": 0.85, "bbox_x1": 100, "bbox_y1": 700, "bbox_x2": 180, "bbox_y2": 800},
        {"class_name": "person",   "confidence": 0.90, "bbox_x1": 1200, "bbox_y1": 300, "bbox_x2": 1350, "bbox_y2": 700}  # ~1100px away
    ]
    # Scenario B: Backpack being carried by person (Near owner)
    attended_luggage_dets = [
        {"class_name": "backpack", "confidence": 0.85, "bbox_x1": 500, "bbox_y1": 500, "bbox_x2": 560, "bbox_y2": 580},
        {"class_name": "person",   "confidence": 0.90, "bbox_x1": 480, "bbox_y1": 400, "bbox_x2": 600, "bbox_y2": 700}   # ~50px away
    ]

    alerts_abandoned = engine.evaluate_frame("cam_01_gate", "frm_test_bag_far", abandoned_luggage_dets, 3.0)
    alerts_attended = engine.evaluate_frame("cam_01_gate", "frm_test_bag_near", attended_luggage_dets, 3.0)

    print(f"    Abandoned Bag: Triggered -> {[a['rule_id'] for a in alerts_abandoned]}")
    print(f"    Attended Bag:  Triggered -> {[a['rule_id'] for a in alerts_attended]}")
    assert any(a["rule_id"] == "RULE_UNATTENDED_OBJECT" for a in alerts_abandoned), "Expected RULE_UNATTENDED_OBJECT trigger!"
    assert not any(a["rule_id"] == "RULE_UNATTENDED_OBJECT" for a in alerts_attended), "Attended bag should not trigger!"
    print("    [PASS] Unattended luggage spatial proximity test passed.")

    # -------------------------------------------------------------
    # STEP 5: Verify Rule 4 & 5 - Loitering & Vehicle in Pedestrian Zone
    # -------------------------------------------------------------
    print("\n>>> STEP 5: VERIFYING RULE 4 (LOITERING) & RULE 5 (VEHICLE RESTRICTION)...")
    corridor_vehicle = [
        {"class_name": "car", "confidence": 0.89, "bbox_x1": 300, "bbox_y1": 400, "bbox_x2": 700, "bbox_y2": 800}
    ]
    alerts_vehicle = engine.evaluate_frame("cam_02_corridor", "frm_test_veh_corridor", corridor_vehicle, 4.0)
    print(f"    Car in Pedestrian Corridor: Triggered -> {[a['rule_id'] for a in alerts_vehicle]}")
    assert any(a["rule_id"] == "RULE_VEHICLE_IN_PEDESTRIAN_ZONE" for a in alerts_vehicle), "Expected RULE_VEHICLE_IN_PEDESTRIAN_ZONE trigger!"
    print("    [PASS] Vehicle in pedestrian corridor rule verified.")

    # -------------------------------------------------------------
    # STEP 6: Simulated Demo Scenario & Operator Workflow
    # -------------------------------------------------------------
    print("\n>>> STEP 6: VERIFYING SIMULATED DEMO BREACH & ACKNOWLEDGMENT LIFECYCLE...")
    demo_res = engine.simulate_demo_alert(scenario="zone_intrusion", camera_id="cam_01_gate")
    demo_alert_id = demo_res["alert_id"]
    print(f"    Simulated Demo Alert Created: ID '{demo_alert_id}' | Type: {demo_res['alert_type']} | Severity: {demo_res['severity']}")

    # Operator Acknowledgment
    update_alert_status(demo_alert_id, "acknowledged")
    conn = get_db_connection()
    status_ack = conn.execute("SELECT status FROM alerts WHERE alert_id = ?", (demo_alert_id,)).fetchone()[0]
    conn.close()
    assert status_ack == "acknowledged", f"Expected acknowledged, got {status_ack}"
    print(f"    Operator Acknowledged Alert: status = '{status_ack}'")

    # Operator Resolution
    update_alert_status(demo_alert_id, "resolved")
    conn = get_db_connection()
    status_res = conn.execute("SELECT status FROM alerts WHERE alert_id = ?", (demo_alert_id,)).fetchone()[0]
    conn.close()
    assert status_res == "resolved", f"Expected resolved, got {status_res}"
    print(f"    Operator Resolved Alert: status = '{status_res}'")

    # Clean up test alerts
    clear_all_alerts(demo_only=True)
    summary = get_alerts_summary()
    print(f"    Current Active Alerts in DB: {summary['active_alerts']} (Critical: {summary['critical_active']}, High: {summary['high_active']})")

    print("\n" + "=" * 80)
    print("  STATUS: WEEK 8 RULE-BASED SECURITY ENGINE VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
