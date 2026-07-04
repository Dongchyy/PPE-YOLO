from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    ROOT
    / "data"
    / "raw"
    / "safe_unsafe_behaviours"
    / "huggingface"
    / "hub"
    / "Voxel51"
    / "Safe_and_Unsafe_Behaviours"
)
RUN_DIR = ROOT / "runs" / "behavior" / "r3d18_clip"


MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def load_samples() -> list[dict]:
    with (DATASET_ROOT / "samples.json").open("r", encoding="utf-8") as f:
        return json.load(f)["samples"]


def select_balanced(samples: list[dict], split: str, per_class: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        if split in (sample.get("tags") or []):
            grouped[sample["ground_truth"]["label"]].append(sample)

    rng = random.Random(seed)
    selected: list[dict] = []
    for label, items in sorted(grouped.items()):
        items = list(items)
        rng.shuffle(items)
        selected.extend(items[: min(per_class, len(items))])
    rng.shuffle(selected)
    return selected


def sample_indices(total_frames: int, clip_len: int, random_clip: bool) -> list[int]:
    if total_frames <= 0:
        return [0] * clip_len
    if total_frames < clip_len:
        return [min(round(i * total_frames / clip_len), total_frames - 1) for i in range(clip_len)]

    if random_clip:
        max_start = max(total_frames - clip_len, 0)
        start = random.randint(0, max_start)
        return list(range(start, start + clip_len))

    start = int(total_frames * 0.1)
    end = max(int(total_frames * 0.9) - 1, start)
    if clip_len == 1:
        return [(start + end) // 2]
    return [round(start + (end - start) * i / (clip_len - 1)) for i in range(clip_len)]


def read_clip(video_path: Path, clip_len: int, image_size: int, train: bool) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(total_frames, clip_len, random_clip=train)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx, 0))
        ok, frame = cap.read()
        if not ok:
            if frames:
                frames.append(frames[-1].clone())
                continue
            else:
                frame = torch.zeros(3, image_size, image_size)
                frames.append(frame)
                continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
        frames.append(tensor)
    cap.release()

    clip = torch.stack(frames, dim=1)
    if train and random.random() < 0.5:
        clip = torch.flip(clip, dims=[3])
    return (clip - MEAN) / STD


class VideoClipDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        label_to_idx: dict[str, int],
        clip_len: int,
        image_size: int,
        train: bool,
    ):
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.clip_len = clip_len
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        video_path = DATASET_ROOT / sample["filepath"]
        label = self.label_to_idx[sample["ground_truth"]["label"]]
        return read_clip(video_path, self.clip_len, self.image_size, self.train), label


def build_model(num_classes: int, freeze_stem: bool) -> nn.Module:
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_stem:
        for name, param in model.named_parameters():
            if not name.startswith("layer4") and not name.startswith("fc"):
                param.requires_grad = False
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    print_every: int,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    start = time.time()

    for step, (clips, labels) in enumerate(loader, start=1):
        clips = clips.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(clips)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * labels.numel()
        preds = logits.argmax(dim=1)
        correct += int(preds.eq(labels).sum().item())
        total += int(labels.numel())

        if train and (step == 1 or step % print_every == 0 or step == len(loader)):
            elapsed = time.time() - start
            print(
                f"epoch {epoch}/{total_epochs} step {step}/{len(loader)} "
                f"loss={total_loss / max(total, 1):.4f} "
                f"acc={correct / max(total, 1):.3f} "
                f"elapsed_min={elapsed / 60:.1f}",
                flush=True,
            )

    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-class", type=int, default=45)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-stem", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    samples = load_samples()
    labels = sorted({s["ground_truth"]["label"] for s in samples})
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_samples = select_balanced(samples, "train", args.train_per_class, args.seed)
    val_samples = select_balanced(samples, "test", args.test_per_class, args.seed)

    train_ds = VideoClipDataset(train_samples, label_to_idx, args.clip_len, args.image_size, train=True)
    val_ds = VideoClipDataset(val_samples, label_to_idx, args.clip_len, args.image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(labels), freeze_stem=args.freeze_stem).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / "classes.json").open("w", encoding="utf-8") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2)

    print(f"device={device}", flush=True)
    print(f"model=r3d_18 total_params={total_params:,} trainable_params={trainable_params:,}", flush=True)
    print(f"clip_shape=Nx3x{args.clip_len}x{args.image_size}x{args.image_size}", flush=True)
    print(f"train_videos={len(train_ds)} val_videos={len(val_ds)} classes={len(labels)}", flush=True)
    print("train videos by class:", dict(Counter(s["ground_truth"]["label"] for s in train_samples)), flush=True)
    print("val videos by class:", dict(Counter(s["ground_truth"]["label"] for s in val_samples)), flush=True)

    metrics_path = RUN_DIR / "metrics.csv"
    best_val_acc = 0.0
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                args.epochs,
                args.print_every,
            )
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                device,
                epoch,
                args.epochs,
                args.print_every,
            )
            best_val_acc = max(best_val_acc, val_acc)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                }
            )
            f.flush()
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "label_to_idx": label_to_idx,
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                RUN_DIR / "last.pt",
            )
            if val_acc >= best_val_acc:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "label_to_idx": label_to_idx,
                        "best_val_acc": best_val_acc,
                        "epoch": epoch,
                    },
                    RUN_DIR / "best.pt",
                )
            print(
                f"epoch {epoch}/{args.epochs} done "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
                f"epoch_min={(time.time() - epoch_start) / 60:.1f}",
                flush=True,
            )

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "label_to_idx": label_to_idx,
            "best_val_acc": best_val_acc,
        },
        RUN_DIR / "r3d18_clip.pt",
    )
    print(f"saved_model={RUN_DIR / 'r3d18_clip.pt'}", flush=True)
    print(f"metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
