from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import yaml


LOG_COLUMNS = [
    "timestamp",
    "session_id",
    "frame_index",
    "track_id",
    "class_name",
    "violation_type",
    "confidence",
    "evidence_path",
]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def ensure_dirs(base_dir: str | Path = ".") -> None:
    """Create project runtime directories."""
    base = Path(base_dir)
    for folder in [
        base / "data" / "sample_videos",
        base / "evidence" / "red_light",
        base / "evidence" / "no_helmet",
        base / "evidence" / "wrong_lane",
        base / "logs",
        base / "models",
        base / "uploads",
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def ensure_log_file(log_path: str | Path) -> None:
    """Keep compatibility for older CSV-based code paths."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        import csv

        csv.writer(file).writerow(LOG_COLUMNS)


def calculate_fps(previous_time: float) -> tuple[float, float]:
    """Calculate FPS from the previous frame timestamp."""
    current_time = time.time()
    elapsed = max(current_time - previous_time, 1e-6)
    return 1.0 / elapsed, current_time


def draw_text_with_background(
    frame,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.6,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
) -> None:
    """Draw readable text on a frame."""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - 6),
        (x + text_w + 8, y + baseline),
        bg_color,
        -1,
    )
    cv2.putText(frame, text, (x + 4, y - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)


def crop_object(frame, bbox: tuple[int, int, int, int]):
    """Crop a bounding box while clamping to frame bounds."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def save_crop(crop, output_path: str | Path) -> str:
    """Save a cropped evidence image and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if crop is None or crop.size == 0:
        return ""

    cv2.imwrite(str(path), crop)
    return str(path)
