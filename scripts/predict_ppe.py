from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


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
    return parser.parse_args()


def class_counts(result) -> Counter[str]:
    counts: Counter[str] = Counter()
    if result.boxes is None:
        return counts

    for cls_id in result.boxes.cls.tolist():
        counts[result.names[int(cls_id)]] += 1
    return counts


def print_safety_rules(counts: Counter[str]) -> None:
    warnings: list[str] = []
    has_person = counts["person"] > 0

    if has_person and counts["glasses"] == 0:
        warnings.append("检测到人员，但未检测到护目镜/眼镜，疑似 PPE 不合规。")
    if counts["hands"] > 0 and counts["gloves"] == 0:
        warnings.append("检测到手部，但未检测到手套，疑似未佩戴手套。")
    if counts["face"] > 0 and counts["face-mask"] == 0:
        warnings.append("检测到面部，但未检测到口罩，疑似未佩戴口罩。")
    if counts["tool"] > 0 and (counts["gloves"] == 0 or counts["glasses"] == 0):
        warnings.append("检测到工具操作相关目标，且关键 PPE 不完整，建议标记为高风险。")

    if warnings:
        print("  安全判断: 异常/需复核")
        for item in warnings:
            print(f"  - {item}")
    else:
        print("  安全判断: 未发现明显 PPE 异常")


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        project=str(ROOT / args.project),
        name=args.name,
    )

    for idx, result in enumerate(results, start=1):
        counts = class_counts(result)
        source_name = Path(result.path).name if result.path else f"result_{idx}"
        print(f"\n{source_name}")
        if counts:
            print("  检测目标:", ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
        else:
            print("  检测目标: 无")

        if not args.no_rules:
            print_safety_rules(counts)

    print(f"\n预测可视化结果已保存到: {ROOT / args.project / args.name}")


if __name__ == "__main__":
    main()
