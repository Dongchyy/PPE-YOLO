import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model = YOLO("yolov8n.pt")
    model.train(
        data="configs/sh17.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project=str(ROOT / "runs" / "detect"),
        name="sh17_yolov8n",
        workers=0,
    )


if __name__ == "__main__":
    main()
