from __future__ import annotations

from core.model_registry import get_yolo_model, resolve_model_path


class ObjectTracker:
    """Track objects with YOLOv8 and ByteTrack."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.35):
        self.model_path = str(resolve_model_path(model_path))
        self.confidence_threshold = confidence_threshold
        self.model = get_yolo_model(self.model_path)
        self.names = self.model.names

    def track(self, frame, classes: list[str] | None = None) -> list[dict]:
        """Return tracked objects with stable track IDs when available."""
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        allowed = set(classes or [])
        tracked_objects: list[dict] = []
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.names.get(class_id, str(class_id))
            if allowed and class_name not in allowed:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            tracked_objects.append(
                {
                    "track_id": int(box.id[0]),
                    "bbox": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "confidence": float(box.conf[0]),
                    "center_point": ((x1 + x2) // 2, (y1 + y2) // 2),
                }
            )
        return tracked_objects
