from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge & build dataset.yaml for UA-DETRAC and Vietnam Traffic Datasets.")
    parser.add_argument("--uadetrac-dir", default="data/processed/ua_detrac_yolo", help="UA-DETRAC dataset directory.")
    parser.add_argument("--vntraffic-dir", default="data/processed/vn_traffic_yolo", help="Vietnam traffic dataset directory.")
    parser.add_argument("--output-yaml", default="data/unified_dataset.yaml", help="Output YAML file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = ROOT_DIR / args.output_yaml if not Path(args.output_yaml).is_absolute() else Path(args.output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_config = {
        "path": str(ROOT_DIR / "data"),
        "train": [
            "processed/ua_detrac_yolo/images/train",
            "processed/vn_traffic_yolo/images/train",
        ],
        "val": [
            "processed/ua_detrac_yolo/images/val",
            "processed/vn_traffic_yolo/images/val",
        ],
        "nc": 4,
        "names": {
            0: "motorcycle",
            1: "car",
            2: "bus",
            3: "truck",
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"Generated unified dataset YAML at: {output_path}")


if __name__ == "__main__":
    main()
