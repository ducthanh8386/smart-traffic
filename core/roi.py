from __future__ import annotations

import cv2
import numpy as np


def create_default_roi(frame_width: int, frame_height: int, roi_ratio: dict | None = None) -> np.ndarray:
    """Create a rectangular ROI from frame-relative ratios."""
    ratio = roi_ratio or {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    x1 = _clamp_ratio_to_pixel(ratio.get("x1", 0.0), frame_width)
    y1 = _clamp_ratio_to_pixel(ratio.get("y1", 0.0), frame_height)
    x2 = _clamp_ratio_to_pixel(ratio.get("x2", 1.0), frame_width)
    y2 = _clamp_ratio_to_pixel(ratio.get("y2", 1.0), frame_height)
    if x2 <= x1:
        x1, x2 = 0, max(frame_width - 1, 0)
    if y2 <= y1:
        y1, y2 = 0, max(frame_height - 1, 0)
    return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.int32)


def _clamp_ratio_to_pixel(value, size: int) -> int:
    ratio = min(max(float(value), 0.0), 1.0)
    return min(max(int(size * ratio), 0), max(size - 1, 0))


def create_default_line(
    frame_width: int,
    frame_height: int,
    line_position_ratio: float = 0.62,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Create a horizontal virtual stop line."""
    y = int(frame_height * line_position_ratio)
    return (int(frame_width * 0.10), y), (int(frame_width * 0.90), y)


def point_in_roi(point: tuple[int, int], roi: np.ndarray) -> bool:
    """Return True when a point is inside the ROI polygon."""
    return cv2.pointPolygonTest(roi, point, False) >= 0


def draw_roi(frame, roi: np.ndarray, color: tuple[int, int, int] = (0, 255, 255)) -> None:
    """Draw the ROI on a frame."""
    cv2.polylines(frame, [roi], isClosed=True, color=color, thickness=2)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi], color=color)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


def draw_line(
    frame,
    line: tuple[tuple[int, int], tuple[int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Draw the virtual stop line."""
    cv2.line(frame, line[0], line[1], color, 3)
