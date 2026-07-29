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


def create_polygon_roi(frame_width: int, frame_height: int, points: list[list[float]] | None = None) -> np.ndarray:
    """Create a custom polygon ROI from pixel points or normalized ratio points."""
    if not points or len(points) < 3:
        return create_default_roi(frame_width, frame_height)

    pts = []
    for pt in points:
        x, y = pt[0], pt[1]
        px = _clamp_ratio_to_pixel(x, frame_width) if isinstance(x, float) and 0.0 <= x <= 1.0 else min(max(int(x), 0), frame_width - 1)
        py = _clamp_ratio_to_pixel(y, frame_height) if isinstance(y, float) and 0.0 <= y <= 1.0 else min(max(int(y), 0), frame_height - 1)
        pts.append((px, py))
    return np.array(pts, dtype=np.int32)


def _clamp_ratio_to_pixel(value, size: int) -> int:
    ratio = min(max(float(value), 0.0), 1.0)
    return min(max(int(size * ratio), 0), max(size - 1, 0))


def create_default_line(
    frame_width: int,
    frame_height: int,
    line_position_ratio: float = 0.62,
    custom_line: list[list[float]] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Create a virtual stop line from explicit points or position ratio."""
    if custom_line and len(custom_line) == 2:
        p1, p2 = custom_line[0], custom_line[1]
        x1 = _clamp_ratio_to_pixel(p1[0], frame_width) if isinstance(p1[0], float) and 0.0 <= p1[0] <= 1.0 else int(p1[0])
        y1 = _clamp_ratio_to_pixel(p1[1], frame_height) if isinstance(p1[1], float) and 0.0 <= p1[1] <= 1.0 else int(p1[1])
        x2 = _clamp_ratio_to_pixel(p2[0], frame_width) if isinstance(p2[0], float) and 0.0 <= p2[0] <= 1.0 else int(p2[0])
        y2 = _clamp_ratio_to_pixel(p2[1], frame_height) if isinstance(p2[1], float) and 0.0 <= p2[1] <= 1.0 else int(p2[1])
        return (x1, y1), (x2, y2)

    y = int(frame_height * line_position_ratio)
    return (int(frame_width * 0.10), y), (int(frame_width * 0.90), y)


def get_perspective_matrix(src_pts: np.ndarray, width: int = 500, height: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Compute Homography Perspective Transform matrix M and inverse M_inv."""
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    src_float = np.float32(src_pts)
    M = cv2.getPerspectiveTransform(src_float, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_float)
    return M, M_inv


def transform_point(point: tuple[int, int], M: np.ndarray) -> tuple[int, int]:
    """Transform point (x, y) using Homography matrix M."""
    px = np.array([[[point[0], point[1]]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(px, M)
    return int(warped[0][0][0]), int(warped[0][0][1])


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
