from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_CLASSES = ("person", "glasses", "gloves", "face-mask", "medical-suit", "safety-suit")
DISPLAY_NAMES = {
    "person": "整个人",
    "glasses": "护目镜/眼镜",
    "gloves": "手套",
    "face-mask": "口罩",
    "medical-suit": "实验服/防护服",
    "safety-suit": "实验服/安全服",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的 YOLO PPE 模型检测图片/视频。")
    parser.add_argument(
        "--weights",
        default="runs/sh17/yolov8n_640_e30_b32/weights/best.pt",
        help="YOLO 权重路径，例如 runs/sh17/.../weights/best.pt",
    )
    parser.add_argument("--source", required=True, help="待预测的图片、视频或文件夹路径。")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值。")
    parser.add_argument("--project", default="runs/predict", help="预测结果保存目录。")
    parser.add_argument("--name", default="ppe", help="本次预测结果子目录名。")
    parser.add_argument("--no-rules", action="store_true", help="只输出检测结果，不输出安全规则判断。")
    parser.add_argument(
        "--target-classes",
        default=",".join(DEFAULT_TARGET_CLASSES),
        help="只检测这些类别，逗号分隔。默认检测整个人、护目镜、手套、口罩、实验服相关类别。",
    )
    parser.add_argument("--all-classes", action="store_true", help="检测 SH17 的全部类别。")
    return parser.parse_args()


def parse_target_classes(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_class_ids(model: YOLO, class_names: list[str]) -> list[int]:
    name_to_id = {name: int(idx) for idx, name in model.names.items()}
    missing = [name for name in class_names if name not in name_to_id]
    if missing:
        available = ", ".join(str(name) for name in model.names.values())
        raise ValueError(f"权重中不存在类别: {missing}. 可用类别: {available}")
    return [name_to_id[name] for name in class_names]


def class_counts(result) -> Counter[str]:
    counts: Counter[str] = Counter()
    if result.boxes is None:
        return counts

    for cls_id in result.boxes.cls.tolist():
        counts[result.names[int(cls_id)]] += 1
    return counts


def print_safety_rules(counts: Counter[str]) -> None:
    warnings: list[str] = []

    suit_count = counts["medical-suit"] + counts["safety-suit"]
    if counts["person"] == 0:
        warnings.append("未检测到人员。")
    if counts["glasses"] == 0:
        warnings.append("未检测到护目镜/眼镜。")
    if counts["gloves"] == 0:
        warnings.append("未检测到手套。")
    if counts["face-mask"] == 0:
        warnings.append("未检测到口罩。")
    if suit_count == 0:
        warnings.append("未检测到实验服/防护服。")

    if warnings:
        print("  安全判断: 异常/需复核")
        for item in warnings:
            print(f"  - {item}")
    else:
        print("  安全判断: 未发现明显 PPE 异常")


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    target_names = parse_target_classes(args.target_classes)
    target_ids = None if args.all_classes else resolve_class_ids(model, target_names)

    if target_ids is not None:
        print("只检测类别:", ", ".join(DISPLAY_NAMES.get(name, name) for name in target_names))

    results = model.predict(
        source=args.source,
        conf=args.conf,
        classes=target_ids,
        save=True,
        project=str(ROOT / args.project),
        name=args.name,
    )

    for idx, result in enumerate(results, start=1):
        counts = class_counts(result)
        source_name = Path(result.path).name if result.path else f"result_{idx}"
        print(f"\n{source_name}")
        if counts:
            print(
                "  检测目标:",
                ", ".join(f"{DISPLAY_NAMES.get(name, name)}={count}" for name, count in sorted(counts.items())),
            )
        else:
            print("  检测目标: 无")

        if not args.no_rules:
            print_safety_rules(counts)

    print(f"\n预测可视化结果已保存到: {ROOT / args.project / args.name}")


if __name__ == "__main__":
    main()
