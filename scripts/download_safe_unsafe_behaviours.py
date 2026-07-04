import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "safe_unsafe_behaviours"
CACHE_DIR = ROOT / ".cache"
FIFTYONE_DIR = ROOT / ".fiftyone"


def configure_environment(endpoint: str | None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIFTYONE_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "huggingface" / "datasets")
    os.environ["FIFTYONE_CONFIG_PATH"] = str(FIFTYONE_DIR / "config.json")
    os.environ["FIFTYONE_APP_CONFIG_PATH"] = str(FIFTYONE_DIR / "app_config.json")
    os.environ["FIFTYONE_DATABASE_DIR"] = str(FIFTYONE_DIR / "db")
    os.environ["FIFTYONE_DEFAULT_DATASET_DIR"] = str(RAW_DIR)
    os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(FIFTYONE_DIR / "zoo")
    os.environ["FIFTYONE_MODEL_ZOO_DIR"] = str(FIFTYONE_DIR / "models")
    os.environ["FIFTYONE_PLUGINS_DIR"] = str(FIFTYONE_DIR / "plugins")
    os.environ["FIFTYONE_DO_NOT_TRACK"] = "true"
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Voxel51/Safe_and_Unsafe_Behaviours with FiftyOne."
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--endpoint", default=None, help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--name", default="safe_unsafe_behaviours")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_environment(args.endpoint)

    from fiftyone.utils.huggingface import load_from_hub

    print(f"raw_dir={RAW_DIR}")
    print(f"hf_home={os.environ['HF_HOME']}")
    print(f"fiftyone_database_dir={os.environ['FIFTYONE_DATABASE_DIR']}")
    if args.endpoint:
        print(f"hf_endpoint={args.endpoint}")

    dataset = load_from_hub(
        "Voxel51/Safe_and_Unsafe_Behaviours",
        name=args.name,
        overwrite=args.overwrite,
        persistent=True,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )

    print(f"dataset_name={dataset.name}")
    print(f"media_type={dataset.media_type}")
    print(f"sample_count={len(dataset)}")
    print(f"default_dataset_dir={RAW_DIR}")
    print("first_samples:")
    for sample in dataset.take(min(5, len(dataset))):
        print(sample.filepath)


if __name__ == "__main__":
    main()
