from __future__ import annotations

from core.model_registry import get_yolo_model, resolve_model_path


class HelmetDetector:
    """Optional detector for future helmet/no-helmet models.

    The MVP works without this model. When a custom model such as
    models/helmet_best.pt exists, pass its path to detect no_helmet boxes.
    """

    def __init__(self, model_path: str | None = None, confidence_threshold: float = 0.35):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.names = {}

        if model_path:
            self.model_path = str(resolve_model_path(model_path))
            self.model = get_yolo_model(self.model_path)
            self.names = self.model.names

    def detect_no_helmet(self, frame) -> list[dict]:
        """Return no_helmet detections when a custom model is loaded."""
        if self.model is None:
            return []

        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        if not results or results[0].boxes is None:
            return []

        detections: list[dict] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.names.get(class_id, str(class_id))
            if class_name != "no_helmet":
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "confidence": float(box.conf[0]),
                }
            )
        return detections
