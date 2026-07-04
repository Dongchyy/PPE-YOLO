from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"
CONFIG_DIR = ROOT / "configs"


def read_split(name: str) -> list[str]:
    split_file = ROOT / f"{name}_files.txt"
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_to_label(image_name: str) -> Path:
    return LABEL_DIR / f"{Path(image_name).stem}.txt"


def write_yolo_list(split: str) -> tuple[int, int, int]:
    names = read_split(split)
    output = CONFIG_DIR / f"sh17_{split}.txt"
    missing_images = 0
    missing_labels = 0
    image_paths: list[str] = []

    for image_name in names:
        image_path = IMAGE_DIR / image_name
        label_path = image_to_label(image_name)
        if not image_path.exists():
            missing_images += 1
            continue
        if not label_path.exists():
            missing_labels += 1
        image_paths.append(image_path.as_posix())

    output.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
    return len(image_paths), missing_images, missing_labels


def main() -> None:
    CONFIG_DIR.mkdir(exist_ok=True)

    image_count = len(list(IMAGE_DIR.glob("*")))
    label_count = len(list(LABEL_DIR.glob("*.txt")))
    print(f"images: {image_count}")
    print(f"labels: {label_count}")

    for split in ("train", "val"):
        total, missing_images, missing_labels = write_yolo_list(split)
        print(
            f"{split}: {total} images, "
            f"{missing_images} missing images, {missing_labels} missing labels"
        )

    print("generated:")
    print(f"  {(CONFIG_DIR / 'sh17_train.txt').as_posix()}")
    print(f"  {(CONFIG_DIR / 'sh17_val.txt').as_posix()}")


if __name__ == "__main__":
    main()
