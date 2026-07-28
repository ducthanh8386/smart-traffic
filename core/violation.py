from datetime import datetime
from pathlib import Path

from core.roi import create_default_roi, point_in_roi
from core.storage import ViolationStorage
import cv2

from core.utils import save_crop


class ViolationDetector:
    """Detect basic traffic violations and write evidence logs."""

    def __init__(
        self,
        storage: ViolationStorage,
        evidence_dir: str | Path = "evidence",
        save_evidence: bool = True,
        crossing_direction: str = "down",
        min_cross_delta_px: int = 2,
    ):
        self.storage = storage
        self.evidence_dir = Path(evidence_dir)
        self.save_evidence = save_evidence
        self.logged_red_light_ids: set[int] = set()
        self.logged_wrong_lane_ids: set[int] = set()
        self.previous_centers: dict[int, tuple[int, int]] = {}
        self.crossing_direction = crossing_direction if crossing_direction in {"down", "up", "both"} else "down"
        self.min_cross_delta_px = max(int(min_cross_delta_px), 0)

    def check_red_light_violation(
        self,
        frame,
        tracked_objects: list[dict],
        line,
        traffic_light: str,
        session_id: str,
        frame_index: int,
    ) -> list[dict]:
        violations = []
        for obj in tracked_objects:
            track_id = int(obj["track_id"])
            crossed = self._crossed_line(track_id, obj["center_point"], line)
            if traffic_light != "RED" or not crossed or track_id in self.logged_red_light_ids:
                continue

            evidence_path = self._save_red_light_evidence(frame, obj, line, session_id, frame_index) if self.save_evidence else ""
            violation = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": session_id,
                "frame_index": int(frame_index),
                "track_id": track_id,
                "class_name": obj["class_name"],
                "violation_type": "red_light_violation",
                "confidence": round(float(obj["confidence"]), 3),
                "evidence_path": evidence_path,
            }
            self.storage.append(violation)
            self.logged_red_light_ids.add(track_id)
            violations.append(violation)
        return violations

    def _crossed_line(self, track_id: int, center: tuple[int, int], line) -> bool:
        line_y = int((line[0][1] + line[1][1]) / 2)
        previous = self.previous_centers.get(track_id)
        self.previous_centers[track_id] = center
        if previous is None:
            return False

        delta = center[1] - previous[1]
        if abs(delta) < self.min_cross_delta_px:
            return False
        crossed_down = previous[1] < line_y <= center[1]
        crossed_up = previous[1] > line_y >= center[1]
        if self.crossing_direction == "down":
            return crossed_down
        if self.crossing_direction == "up":
            return crossed_up
        return crossed_down or crossed_up

    def _save_red_light_evidence(self, frame, obj: dict, line, session_id: str, frame_index: int) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_session = "".join(ch for ch in session_id if ch.isalnum())[:16] or "session"
        path = (
            self.evidence_dir
            / "red_light"
            / f"{safe_session}_frame_{int(frame_index)}_track_{obj['track_id']}_{timestamp}.jpg"
        )
        evidence_frame = frame.copy()
        x1, y1, x2, y2 = obj["bbox"]
        cv2.line(evidence_frame, line[0], line[1], (0, 0, 255), 3)
        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"RED LIGHT | {obj['class_name']} ID:{obj['track_id']} {obj['confidence']:.2f}"
        cv2.putText(
            evidence_frame,
            label,
            (max(x1, 10), max(y1 - 10, 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            evidence_frame,
            label,
            (max(x1, 10), max(y1 - 10, 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            evidence_frame,
            f"Frame {int(frame_index)}",
            (10, evidence_frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            evidence_frame,
            f"Frame {int(frame_index)}",
            (10, evidence_frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        saved_path = save_crop(evidence_frame, path)
        if not saved_path:
            return ""
        return f"/api/evidence/{Path(saved_path).resolve().relative_to(self.evidence_dir.resolve()).as_posix()}"

    def check_wrong_lane_violation(
        self,
        frame,
        tracked_objects: list[dict],
        lanes_config: list[dict],
        frame_width: int,
        frame_height: int,
        session_id: str,
        frame_index: int,
    ) -> list[dict]:
        violations = []
        for obj in tracked_objects:
            track_id = int(obj["track_id"])
            if track_id in self.logged_wrong_lane_ids:
                continue

            center = obj["center_point"]
            for lane in lanes_config:
                lane_roi = create_default_roi(frame_width, frame_height, lane.get("roi_ratio"))
                if point_in_roi(center, lane_roi):
                    allowed = lane.get("allowed_classes", [])
                    if obj["class_name"] not in allowed:
                        evidence_path = self._save_wrong_lane_evidence(frame, obj, lane_roi, lane["name"], session_id, frame_index) if self.save_evidence else ""
                        violation = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "session_id": session_id,
                            "frame_index": int(frame_index),
                            "track_id": track_id,
                            "class_name": obj["class_name"],
                            "violation_type": "wrong_lane_violation",
                            "confidence": round(float(obj["confidence"]), 3),
                            "evidence_path": evidence_path,
                        }
                        self.storage.append(violation)
                        self.logged_wrong_lane_ids.add(track_id)
                        violations.append(violation)
                        break
        return violations

    def _save_wrong_lane_evidence(self, frame, obj: dict, lane_roi, lane_name: str, session_id: str, frame_index: int) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_session = "".join(ch for ch in session_id if ch.isalnum())[:16] or "session"
        path = (
            self.evidence_dir
            / "wrong_lane"
            / f"{safe_session}_frame_{int(frame_index)}_track_{obj['track_id']}_{timestamp}.jpg"
        )
        evidence_frame = frame.copy()
        x1, y1, x2, y2 = obj["bbox"]
        cv2.polylines(evidence_frame, [lane_roi], isClosed=True, color=(0, 0, 255), thickness=3)
        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"WRONG LANE ({lane_name}) | {obj['class_name']} ID:{obj['track_id']} {obj['confidence']:.2f}"
        cv2.putText(
            evidence_frame,
            label,
            (max(x1, 10), max(y1 - 10, 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            evidence_frame,
            label,
            (max(x1, 10), max(y1 - 10, 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            evidence_frame,
            f"Frame {int(frame_index)}",
            (10, evidence_frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            evidence_frame,
            f"Frame {int(frame_index)}",
            (10, evidence_frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        saved_path = save_crop(evidence_frame, path)
        if not saved_path:
            return ""
        return f"/api/evidence/{Path(saved_path).resolve().relative_to(self.evidence_dir.resolve()).as_posix()}"

