from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Standardized SmartTraffic taxonomy
CLASS_MAPPING = {
    "car": 1,
    "van": 1,
    "bus": 2,
    "others": 3,  # Truck/others mapped to truck
    "truck": 3,
    "motorcycle": 0,
    "xe_may": 0,
    "o_to": 1,
    "xe_bus": 2,
    "xe_tai": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert UA-DETRAC XML annotations to YOLO format.")
    parser.add_argument("--xml-dir", required=True, help="Directory containing UA-DETRAC XML files.")
    parser.add_argument("--output-dir", default="data/processed/ua_detrac_yolo", help="Output directory for YOLO labels.")
    parser.add_argument("--img-width", type=int, default=960, help="Standard width of frames.")
    parser.add_argument("--img-height", type=int, default=540, help="Standard height of frames.")
    return parser.parse_args()


def convert_uadetrac_xml(xml_path: Path, output_dir: Path, img_width: int, img_height: int) -> int:
    """Convert a single UA-DETRAC XML file into YOLO label txt files per frame."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seq_name = xml_path.stem
    seq_out_dir = output_dir / seq_name
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    converted_frames = 0
    for frame in root.findall("frame"):
        frame_num = int(frame.get("num", 0))
        target_list = frame.find("target_list")
        if target_list is None:
            continue

        label_file = seq_out_dir / f"img{frame_num:05d}.txt"
        lines = []
        for target in target_list.findall("target"):
            box = target.find("box")
            attribute = target.find("attribute")
            if box is None:
                continue

            left = float(box.get("left", 0))
            top = float(box.get("top", 0))
            width = float(box.get("width", 0))
            height = float(box.get("height", 0))

            vehicle_type = "car"
            if attribute is not None and "vehicle_type" in attribute.attrib:
                vehicle_type = attribute.get("vehicle_type", "car").lower()

            class_id = CLASS_MAPPING.get(vehicle_type, 1)

            # Convert to YOLO normalized center format
            x_center = (left + width / 2.0) / img_width
            y_center = (top + height / 2.0) / img_height
            w_norm = width / img_width
            h_norm = height / img_height

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        with label_file.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        converted_frames += 1

    return converted_frames


def main() -> None:
    args = parse_args()
    xml_dir = Path(args.xml_dir)
    output_dir = ROOT_DIR / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    if not xml_dir.exists():
        print(f"Error: XML directory not found: {xml_dir}")
        sys.exit(1)

    xml_files = list(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        sys.exit(1)

    total_frames = 0
    for xml_file in xml_files:
        frames = convert_uadetrac_xml(xml_file, output_dir, args.img_width, args.img_height)
        total_frames += frames
        print(f"Converted {xml_file.name} ({frames} frames) -> {output_dir / xml_file.stem}")

    print(f"\nDone! Converted {len(xml_files)} sequences with total {total_frames} frame label files.")


if __name__ == "__main__":
    main()
