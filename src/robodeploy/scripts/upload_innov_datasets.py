"""上传本地 LeRobot 数据集到 Hugging Face Hub 的同一仓库子目录下."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "LiFenrir/invrobot"

DATASET_ROOT = Path("/home/kemove/INNOV/datasets/innov_arm")

# 要上传的数据集目录名，远程子目录与本地同名
DATASETS = [
    "innov_0730_0731_3cam",
    "innov_0730_0731_3cam_clean",
    "innov_0730_0731_4cam",
    "innov_0730_0731_4cam_clean",
]


def upload_dataset(api: HfApi, local_root: str | Path, path_in_repo: str) -> None:
    """将本地数据集目录上传到仓库的指定子目录."""
    local_root = Path(local_root)
    print(f"\n{'=' * 60}")
    print(f"Local: {local_root}")
    print(f"Remote: {REPO_ID}/{path_in_repo}")
    print("Uploading...")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=local_root,
        path_in_repo=path_in_repo,
        ignore_patterns=["images/"],
    )
    print(f"Done: https://huggingface.co/datasets/{REPO_ID}/tree/main/{path_in_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload INNOV datasets to Hugging Face Hub subdirectories.")
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (or set HF_TOKEN env var).",
    )
    parser.add_argument("--private", action="store_true", help="Create private dataset repo.")
    args = parser.parse_args()

    if not args.token:
        raise ValueError("Please provide --token or set HF_TOKEN environment variable.")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=args.private, exist_ok=True)

    for name in DATASETS:
        local_root = DATASET_ROOT / name
        if not local_root.exists():
            raise FileNotFoundError(f"数据集不存在: {local_root}")
        upload_dataset(api, local_root, name)

    print(f"\nAll uploaded to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
