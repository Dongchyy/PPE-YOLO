from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models.video import r3d_18

from train_behavior_r3d18 import read_clip


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的 R3D-18 模型预测单个行为视频。")
    parser.add_argument(
        "--weights",
        default="runs/behavior/r3d18_clip/best.pt",
        help="行为识别权重路径，例如 runs/behavior/r3d18_clip/best.pt",
    )
    parser.add_argument("--source", required=True, help="待预测的 mp4 视频路径。")
    parser.add_argument("--clip-len", type=int, default=None, help="抽取帧数，默认读取训练参数。")
    parser.add_argument("--image-size", type=int, default=None, help="输入尺寸，默认读取训练参数。")
    parser.add_argument("--topk", type=int, default=3, help="输出概率最高的类别数量。")
    return parser.parse_args()


def build_model(num_classes: int) -> nn.Module:
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_label_names(checkpoint: dict) -> list[str]:
    label_to_idx = checkpoint["label_to_idx"]
    labels = [None] * len(label_to_idx)
    for label, idx in label_to_idx.items():
        labels[int(idx)] = label
    return [str(label) for label in labels]


def main() -> None:
    args = parse_args()
    weights_path = ROOT / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    source_path = ROOT / args.source if not Path(args.source).is_absolute() else Path(args.source)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device)
    labels = load_label_names(checkpoint)
    train_args = checkpoint.get("args", {})
    clip_len = args.clip_len or int(train_args.get("clip_len", 16))
    image_size = args.image_size or int(train_args.get("image_size", 112))

    model = build_model(len(labels)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    clip = read_clip(source_path, clip_len=clip_len, image_size=image_size, train=False)
    clip = clip.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(clip)
        probs = logits.softmax(dim=1).squeeze(0).cpu()

    topk = min(args.topk, len(labels))
    values, indices = torch.topk(probs, k=topk)

    print(f"video={source_path}")
    print(f"weights={weights_path}")
    print(f"device={device}")
    print(f"clip_shape=1x3x{clip_len}x{image_size}x{image_size}")
    print("top predictions:")
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"  {rank}. {labels[idx]}: {score:.4f}")

    output = {
        "video": str(source_path),
        "weights": str(weights_path),
        "clip_len": clip_len,
        "image_size": image_size,
        "predictions": [
            {"label": labels[int(idx)], "score": float(score)}
            for score, idx in zip(values.tolist(), indices.tolist())
        ],
    }
    print("json=" + json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
