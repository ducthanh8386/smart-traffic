from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock

from core.runtime import configure_runtime

configure_runtime()


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
BUILTIN_MODELS = {"yolov8n.pt", "yolov8s.pt"}

_model_lock = Lock()


def resolve_model_path(model_path: str | None) -> Path:
    """Resolve a user-selected model to an allowed local .pt file."""
    requested = (model_path or "yolov8n.pt").strip() or "yolov8n.pt"
    candidate = Path(requested)

    if candidate.name in BUILTIN_MODELS and len(candidate.parts) == 1:
        resolved = (ROOT_DIR / candidate.name).resolve()
    else:
        if candidate.suffix.lower() != ".pt":
            raise ValueError("Model must be a .pt file.")
        if candidate.is_absolute():
            raise ValueError("Absolute model paths are not allowed.")

        models_root = MODELS_DIR.resolve()
        resolved = (ROOT_DIR / candidate).resolve()
        if models_root not in resolved.parents:
            raise ValueError("Custom models must be stored under models/.")

    if not resolved.exists() or not resolved.is_file():
        raise ValueError("Selected model file does not exist.")
    return resolved


def to_project_model_path(resolved_model_path: Path) -> str:
    resolved = resolved_model_path.resolve()
    if resolved.parent == ROOT_DIR:
        return resolved.name
    return resolved.relative_to(ROOT_DIR).as_posix()


def list_available_models() -> list[str]:
    """List built-in and models/*.pt files selectable by the frontend."""
    models: set[str] = set()
    for name in sorted(BUILTIN_MODELS):
        path = ROOT_DIR / name
        if path.is_file():
            models.add(name)
    if MODELS_DIR.exists():
        for path in sorted(MODELS_DIR.glob("*.pt")):
            if path.is_file():
                models.add(to_project_model_path(path))
    return sorted(models)


@lru_cache(maxsize=4)
def _load_model_cached(path: str):
    from ultralytics import YOLO

    return YOLO(path)


def get_yolo_model(model_path: str | Path):
    """Return a cached YOLO instance for the resolved model path."""
    resolved = str(Path(model_path).resolve())
    with _model_lock:
        return _load_model_cached(resolved)
