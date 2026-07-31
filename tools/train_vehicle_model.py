from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.runtime import configure_runtime

configure_runtime()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom YOLO vehicle detector.")
    parser.add_argument("--data", help="Path to YOLO dataset.yaml. If omitted, the script searches under data/.")
    parser.add_argument("--base-model", default="yolov8s.pt", help="Base model, e.g. yolov8s.pt or yolov8m.pt.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 or 0,1 or cpu.")
    parser.add_argument("--name", default="smarttraffic_vehicle")
    parser.add_argument("--output-model", default="models/vehicle_best.pt", help="Where to copy the trained best.pt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_dataset_yaml(args.data)
    output_model = resolve_output_model(args.output_model)

    from ultralytics import YOLO

    model = YOLO(str(resolve_base_model(args.base_model)))
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT_DIR / "runs"),
        name=args.name,
    )
    best_path = find_best_model_path(args.name, results)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, output_model)
    print(f"Training done. Model copied to {output_model.relative_to(ROOT_DIR).as_posix()}")


def resolve_base_model(base_model: str) -> Path | str:
    candidate = Path(base_model)
    if candidate.is_absolute():
        return candidate
    local = ROOT_DIR / candidate
    return local if local.exists() else base_model


def resolve_dataset_yaml(data_arg: str | None) -> Path:
    if data_arg:
        path = Path(data_arg)
        if not path.is_absolute():
            path = ROOT_DIR / path
        path = path.resolve()
        if looks_like_placeholder(path):
            raise SystemExit(f"Dataset path is still a placeholder: {path}")
        if not path.is_file():
            raise SystemExit(f"Dataset YAML not found: {path}")
        return path

    candidates = find_dataset_yamls()
    if len(candidates) == 1:
        print(f"Using detected dataset: {candidates[0]}")
        return candidates[0]
    if candidates:
        joined = "\n".join(f"  - {path}" for path in candidates)
        raise SystemExit(f"Multiple dataset YAML files found. Pass one with --data:\n{joined}")
    raise SystemExit(
        "No YOLO dataset.yaml found under data/.\n"
        "Create a labeled YOLO dataset first, then run for example:\n"
        "  python tools\\train_vehicle_model.py --data data\\vehicle_dataset\\dataset.yaml"
    )


def find_dataset_yamls() -> list[Path]:
    roots = [ROOT_DIR / "data", ROOT_DIR / "datasets"]
    names = {"dataset.yaml", "dataset.yml", "data.yaml", "data.yml"}
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower() in names:
                found.append(path.resolve())
    return sorted(found)


def looks_like_placeholder(path: Path) -> bool:
    normalized = str(path).lower()
    return "duong-dan-that" in normalized or "\\path\\to\\" in normalized or "/path/to/" in normalized


def resolve_output_model(output_model: str) -> Path:
    path = Path(output_model)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if path.suffix.lower() != ".pt":
        raise SystemExit("--output-model must end with .pt")
    return path.resolve()


def find_best_model_path(run_name: str, results) -> Path:
    save_dir = getattr(results, "save_dir", None)
    candidates: list[Path] = []
    if save_dir:
        candidates.append(Path(save_dir) / "weights" / "best.pt")
    candidates.extend((ROOT_DIR / "runs").rglob("best.pt"))
    candidates = [path.resolve() for path in candidates if path.is_file()]
    if not candidates:
        raise SystemExit("Training finished, but best.pt was not found under runs/.")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if run_name in str(path):
            return path
    return candidates[0]


if __name__ == "__main__":
    main()
