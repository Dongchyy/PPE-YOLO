import argparse
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPE detection with a trained YOLO model.")
    parser.add_argument("--weights", default="runs/detect/sh17_yolov8n/weights/best.pt")
    parser.add_argument("--source", default="images")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        project=str(ROOT / "runs" / "predict"),
        name="ppe",
    )


if __name__ == "__main__":
    main()
