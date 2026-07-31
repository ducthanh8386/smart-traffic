from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.runtime import configure_runtime

configure_runtime()

# Mapping project class names and IDs
CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# COCO dataset vehicle IDs to project class IDs
# COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
COCO_TO_PROJECT = {
    2: 0,  # car
    3: 1,  # motorcycle
    5: 2,  # bus
    7: 3,  # truck
}

FOLDER_ALIASES = {
    "car": "car",
    "cars": "car",
    "oto": "car",
    "o_to": "car",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "xe_may": "motorcycle",
    "xemay": "motorcycle",
    "bus": "bus",
    "buses": "bus",
    "xe_buyt": "bus",
    "truck": "truck",
    "trucks": "truck",
    "xe_tai": "truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label raw vehicle images using AI and build YOLO dataset.")
    parser.add_argument("--raw-dir", default="data/raw_images", help="Directory containing folders of raw images.")
    parser.add_argument("--output-dir", default="data/vehicle_dataset", help="Output YOLO dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio (0.0 to 1.0).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for auto-annotation.")
    parser.add_argument("--model", default="yolov8s.pt", help="Pre-trained base model for labeling.")
    return parser.parse_args()


def get_target_class_id(folder_name: str) -> int | None:
    norm = folder_name.lower().strip()
    target_name = FOLDER_ALIASES.get(norm)
    if target_name:
        return CLASS_TO_ID[target_name]
    for key, name in FOLDER_ALIASES.items():
        if key in norm:
            return CLASS_TO_ID[name]
    return None


def main() -> None:
    args = parse_args()
    raw_dir = ROOT_DIR / args.raw_dir if not Path(args.raw_dir).is_absolute() else Path(args.raw_dir)
    output_dir = ROOT_DIR / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    if not raw_dir.exists():
        print(f"Error: Raw images directory does not exist: {raw_dir}")
        print(f"Please create folders like:\n  {raw_dir / 'car'}\n  {raw_dir / 'motorcycle'}\n  {raw_dir / 'bus'}\n  {raw_dir / 'truck'}")
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder_files: list[tuple[Path, int | None]] = []

    for item in raw_dir.iterdir():
        if item.is_dir():
            target_id = get_target_class_id(item.name)
            for file_path in item.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    folder_files.append((file_path, target_id))

    # Also check root of raw_dir if any images exist
    for file_path in raw_dir.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            folder_files.append((file_path, None))

    if not folder_files:
        print(f"No image files found under {raw_dir}")
        sys.exit(1)

    print(f"Found {len(folder_files)} images. Loading AI model ({args.model}) for auto-labeling...")

    from ultralytics import YOLO

    model = YOLO(args.model)

    # Prepare output folders
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(folder_files)

    num_val = int(len(folder_files) * args.val_ratio)
    val_files = set(folder_files[:num_val])

    labeled_count = 0
    fallback_count = 0

    for idx, (img_path, folder_class_id) in enumerate(folder_files, 1):
        is_val = (img_path, folder_class_id) in val_files
        dest_img_dir = val_img_dir if is_val else train_img_dir
        dest_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        stem = f"img_{idx:05d}_{img_path.stem}"
        target_img_path = dest_img_dir / f"{stem}{img_path.suffix.lower()}"
        target_lbl_path = dest_lbl_dir / f"{stem}.txt"

        # Copy image
        shutil.copy2(img_path, target_img_path)

        # Run AI detection
        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        boxes_lines = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                coco_cls = int(box.cls[0].item())
                if coco_cls in COCO_TO_PROJECT:
                    proj_cls = COCO_TO_PROJECT[coco_cls]
                    # xywhn normalized
                    xywhn = box.xywhn[0].tolist()
                    x, y, w, h = xywhn
                    boxes_lines.append(f"{proj_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        if boxes_lines:
            labeled_count += 1
        else:
            # Fallback if AI didn't detect vehicle but we know the folder class
            if folder_class_id is not None:
                # Default box covering central region of image
                boxes_lines.append(f"{folder_class_id} 0.500000 0.500000 0.900000 0.900000")
                fallback_count += 1

        with target_lbl_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(boxes_lines) + "\n" if boxes_lines else "")

        if idx % 10 == 0 or idx == len(folder_files):
            print(f"Processed [{idx}/{len(folder_files)}] images...")

    # Write dataset.yaml
    yaml_content = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
"""
    yaml_path = output_dir / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n" + "=" * 50)
    print("AUTO-LABELING & DATASET PREPARATION COMPLETED!")
    print(f"Total images processed: {len(folder_files)}")
    print(f"Images labeled with AI: {labeled_count}")
    print(f"Fallback labeled: {fallback_count}")
    print(f"Dataset YAML generated at: {yaml_path}")
    print("=" * 50)
    print("\nYou can now start training by running:")
    print("python tools\\train_vehicle_model.py --data data\\vehicle_dataset\\dataset.yaml")


if __name__ == "__main__":
    main()
