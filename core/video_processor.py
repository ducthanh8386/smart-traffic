from __future__ import annotations

import time
from collections import Counter
from typing import Any

import cv2

from core.density import DensityEstimator
from core.helmet_detector import HelmetDetector
from core.roi import create_default_line, create_default_roi, create_polygon_roi, draw_line, draw_roi
from core.storage import get_violation_storage
from core.tracker import ObjectTracker
from core.traffic_light_detector import TrafficLightDetector
from core.utils import calculate_fps, draw_text_with_background
from core.violation import ViolationDetector


class VideoProcessor:
    """Coordinate all computer-vision steps for one video stream."""

    TRACK_CLASSES = ["car", "motorcycle", "bus", "truck", "person"]
    VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

    def __init__(
        self,
        config: dict[str, Any],
        model_path: str,
        traffic_light: str = "RED",
        max_capacity: int = 30,
        show_boxes: bool = True,
        show_roi: bool = True,
        show_line: bool = True,
        show_lanes: bool = False,
        save_evidence: bool = True,
        target_classes: list[str] | None = None,
    ):
        self.config = config
        self.traffic_light = traffic_light
        self.show_boxes = show_boxes
        self.show_roi = show_roi
        self.show_line = show_line
        self.show_lanes = show_lanes
        self.target_classes = target_classes if target_classes is not None else self.TRACK_CLASSES
        self.previous_time = time.time()

        confidence = float(config.get("confidence_threshold", 0.35))
        thresholds = config.get("density_threshold", {})

        self.tracker = ObjectTracker(model_path=model_path, confidence_threshold=confidence)
        self.density_estimator = DensityEstimator(
            max_capacity=max_capacity,
            normal_threshold=thresholds.get("normal", 40),
            crowded_threshold=thresholds.get("crowded", 70),
        )
        self.violation_detector = ViolationDetector(
            storage=get_violation_storage(config.get("violation_db_path", "logs/violations.sqlite3")),
            evidence_dir=config.get("evidence_dir", "evidence"),
            save_evidence=save_evidence,
            crossing_direction=config.get("line_crossing_direction", "down"),
        )
        self.helmet_detector = HelmetDetector(confidence_threshold=confidence)
        self.traffic_light_detector = TrafficLightDetector()

    def process_frame(self, frame, session_id: str = "", frame_index: int = 0) -> tuple[Any, dict[str, Any]]:
        """Process and annotate a single BGR frame."""
        frame_height, frame_width = frame.shape[:2]
        custom_roi = self.config.get("custom_roi_points")
        if custom_roi:
            roi = create_polygon_roi(frame_width, frame_height, custom_roi)
        else:
            roi = create_default_roi(frame_width, frame_height, self.config.get("roi_ratio"))

        custom_line = self.config.get("custom_line_points")
        line = create_default_line(
            frame_width,
            frame_height,
            float(self.config.get("line_position_ratio", 0.62)),
            custom_line=custom_line,
        )

        effective_light = self.traffic_light
        if self.traffic_light == "AUTO":
            detected_light = self.traffic_light_detector.detect_state(frame, roi)
            effective_light = detected_light if detected_light != "UNKNOWN" else "RED"

        tracked_objects = self.tracker.track(frame, classes=self.target_classes)
        vehicle_count_roi, vehicles_in_roi = self.density_estimator.count_vehicles_in_roi(tracked_objects, roi)
        density_percent = self.density_estimator.calculate_density_percent(vehicle_count_roi)
        pcu_metrics = self.density_estimator.analyze_pcu_metrics(vehicles_in_roi)
        pcu_density_percent = pcu_metrics["pcu_density_percent"]
        traffic_status = self.density_estimator.get_traffic_status(max(density_percent, pcu_density_percent))
        recommendation = self.density_estimator.get_recommendation(traffic_status)

        red_light_violations = []
        if effective_light != "NONE":
            red_light_violations = self.violation_detector.check_red_light_violation(
                frame,
                vehicles_in_roi,
                line,
                effective_light,
                session_id=session_id,
                frame_index=frame_index,
            )

        wrong_lane_violations = []
        if self.show_lanes:
            wrong_lane_violations = self.violation_detector.check_wrong_lane_violation(
                frame,
                tracked_objects,
                self.config.get("lanes", []),
                frame_width,
                frame_height,
                session_id=session_id,
                frame_index=frame_index,
            )
        violations = red_light_violations + wrong_lane_violations

        fps, self.previous_time = calculate_fps(self.previous_time)
        class_counts = Counter(obj["class_name"] for obj in tracked_objects if obj["class_name"] in self.VEHICLE_CLASSES)

        output_frame = frame.copy()
        self._draw_frame_overlay(output_frame, tracked_objects, roi, line, fps, vehicle_count_roi, density_percent, traffic_status)

        return output_frame, {
            "fps": round(fps, 2),
            "total_current_vehicles": sum(class_counts.values()),
            "car": class_counts.get("car", 0),
            "motorcycle": class_counts.get("motorcycle", 0),
            "bus": class_counts.get("bus", 0),
            "truck": class_counts.get("truck", 0),
            "vehicle_count_roi": vehicle_count_roi,
            "density_percent": round(density_percent, 2),
            "pcu_total": pcu_metrics["pcu_total"],
            "pcu_density_percent": pcu_density_percent,
            "motorcycle_ratio_percent": pcu_metrics["motorcycle_ratio_percent"],
            "traffic_status": traffic_status,
            "recommendation": recommendation,
            "traffic_light": effective_light,
            "violations": violations,
        }

    def _draw_frame_overlay(
        self,
        frame,
        tracked_objects: list[dict],
        roi,
        line,
        fps: float,
        vehicle_count_roi: int,
        density_percent: float,
        traffic_status: str,
    ) -> None:
        if self.show_roi:
            draw_roi(frame, roi)
            if self.show_lanes:
                self._draw_lanes(frame)
        if self.show_line:
            if self.traffic_light == "RED":
                line_color = (0, 0, 255)
            elif self.traffic_light == "YELLOW":
                line_color = (0, 255, 255)
            elif self.traffic_light == "GREEN":
                line_color = (0, 255, 0)
            else:
                line_color = (160, 160, 160)
            draw_line(frame, line, color=line_color)
        if self.show_boxes:
            self._draw_objects(frame, tracked_objects)

        draw_text_with_background(frame, f"Light: {self.traffic_light}", (12, 30), bg_color=(0, 0, 180))
        draw_text_with_background(frame, f"FPS: {fps:.1f}", (12, 60))
        draw_text_with_background(frame, f"ROI vehicles: {vehicle_count_roi}", (12, 90))
        draw_text_with_background(frame, f"Density: {density_percent:.1f}%", (12, 120))
        draw_text_with_background(frame, f"Status: {traffic_status}", (12, 150))

    def _draw_objects(self, frame, tracked_objects: list[dict]) -> None:
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            class_name = obj["class_name"]
            color = (0, 200, 0) if class_name != "person" else (255, 160, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} ID:{obj['track_id']} {obj['confidence']:.2f}"
            draw_text_with_background(frame, label, (x1, max(y1, 24)), bg_color=color)
            cv2.circle(frame, obj["center_point"], 4, (255, 255, 255), -1)

    def _draw_lanes(self, frame) -> None:
        frame_height, frame_width = frame.shape[:2]
        lanes = self.config.get("lanes", [])
        for i, lane in enumerate(lanes):
            lane_roi = create_default_roi(frame_width, frame_height, lane.get("roi_ratio"))
            color = (255, 128, 0) if i % 2 == 0 else (255, 0, 128)
            cv2.polylines(frame, [lane_roi], isClosed=True, color=color, thickness=2)
            # Label the lane at the top-center of the lane region
            x1, y1 = lane_roi[0]
            x2, y2 = lane_roi[2]
            text_x = (x1 + x2) // 2 - 40
            text_y = y1 + 30
            draw_text_with_background(frame, lane["name"], (max(int(text_x), 10), max(int(text_y), 30)), bg_color=color)

