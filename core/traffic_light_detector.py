from __future__ import annotations

import cv2
import numpy as np


class TrafficLightDetector:
    """Detect traffic light state (RED, GREEN, YELLOW) using Computer Vision HSV & contour analysis."""

    def __init__(self, min_area: int = 15, min_circularity: float = 0.5):
        self.min_area = min_area
        self.min_circularity = min_circularity

    def detect_state(self, frame: np.ndarray, light_roi: np.ndarray | None = None) -> str:
        """Detect dominant traffic light state in frame or light ROI."""
        if frame is None or frame.size == 0:
            return "UNKNOWN"

        target_region = frame
        if light_roi is not None and len(light_roi) == 4:
            x1, y1 = np.min(light_roi, axis=0)
            x2, y2 = np.max(light_roi, axis=0)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            if x2 > x1 and y2 > y1:
                target_region = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(target_region, cv2.COLOR_BGR2HSV)

        # Red ranges (wraps around 0/180 in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Yellow range
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])

        # Green range
        lower_green = np.array([40, 90, 90])
        upper_green = np.array([90, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        red_score = self._evaluate_signal_mask(mask_red)
        yellow_score = self._evaluate_signal_mask(mask_yellow)
        green_score = self._evaluate_signal_mask(mask_green)

        scores = {"RED": red_score, "YELLOW": yellow_score, "GREEN": green_score}
        best_state = max(scores, key=scores.get)

        if scores[best_state] <= 0:
            return "UNKNOWN"
        return best_state

    def _evaluate_signal_mask(self, mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_score = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity >= self.min_circularity:
                total_score += area * circularity
            else:
                total_score += area * 0.5
        return total_score
