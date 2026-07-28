import pytest

from core.model_registry import MODELS_DIR, ROOT_DIR, list_available_models, resolve_model_path, to_project_model_path


def test_resolve_builtin_model() -> None:
    assert resolve_model_path("yolov8n.pt") == (ROOT_DIR / "yolov8n.pt").resolve()


def test_rejects_path_traversal() -> None:
    with pytest.raises(ValueError):
        resolve_model_path("../yolov8n.pt")


def test_rejects_absolute_custom_model_path() -> None:
    with pytest.raises(ValueError):
        resolve_model_path(str(MODELS_DIR / "unit_test_model.pt"))


def test_allows_pt_inside_models() -> None:
    model_path = MODELS_DIR / "unit_test_model.pt"
    model_path.write_bytes(b"test")
    try:
        assert resolve_model_path("models/unit_test_model.pt") == model_path.resolve()
        assert to_project_model_path(model_path) == "models/unit_test_model.pt"
        assert "models/unit_test_model.pt" in list_available_models()
    finally:
        model_path.unlink(missing_ok=True)
