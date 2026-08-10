#!/usr/bin/env python3
"""提取 LeRobot 数据集第一个 episode 的第一帧图像并保存为图片。

支持 video/image 两种存储类型的视觉模态。默认保存所有 camera 的第一帧。
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from robodeploy.datasets.lerobot_dataset import LeRobotDataset


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """将 (C,H,W) 或 (H,W) 的 float32 [0,1] tensor 转成 uint8 numpy 数组。"""
    if tensor.ndim == 3:
        # (C,H,W) -> (H,W,C)
        img = tensor.permute(1, 2, 0).cpu().numpy()
        if img.shape[2] == 1:
            img = img.squeeze(2)
    elif tensor.ndim == 2:
        img = tensor.cpu().numpy()
    else:
        raise ValueError(f"不支持的图像维度: {tensor.shape}")

    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def extract_first_frame(
    dataset_root: str | Path,
    output_dir: str | Path | None = None,
    camera: str | None = None,
):
    root = Path(dataset_root).resolve()
    out_dir = root / "first_frame" if output_dir is None else Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (root / "meta" / "info.json").is_file():
        raise ValueError(f"数据集目录无效，缺少 meta/info.json: {root}")

    dataset = LeRobotDataset(repo_id=root.name, root=root, download_videos=False)
    camera_keys = dataset.meta.camera_keys
    if not camera_keys:
        raise ValueError("数据集未找到任何图像/视频模态")

    first_frame = dataset[0]
    cameras_to_save = [camera] if camera else camera_keys

    for cam in cameras_to_save:
        if cam not in camera_keys:
            raise ValueError(f"摄像头 '{cam}' 不存在，可用: {camera_keys}")

        img_tensor = first_frame[cam]  # (C,H,W) float32，范围 [0,1]
        img_array = tensor_to_image(img_tensor)
        safe_name = cam.replace(".", "_").replace("/", "_")
        out_path = out_dir / f"{safe_name}_first_frame.png"
        Image.fromarray(img_array).save(out_path)
        print(f"已保存: {out_path}  shape={img_array.shape}")


def main():
    parser = argparse.ArgumentParser(description="提取 LeRobot 数据集第一帧图像")
    parser.add_argument("dataset_root", type=str, help="数据集根目录路径")
    parser.add_argument("--output-dir", type=str, default=None, help="图片保存目录（默认数据集根目录下的 first_frame/）")
    parser.add_argument("--camera", type=str, default=None, help="只保存指定 camera key（默认全部）")
    args = parser.parse_args()

    extract_first_frame(args.dataset_root, args.output_dir, args.camera)


if __name__ == "__main__":
    main()
