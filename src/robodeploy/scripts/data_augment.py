#!/usr/bin/env python3
"""
Full data augmentation pipeline for LeRobot bi_s1 datasets.

Pipeline:
  1. Stack front + front_1 (front_1 rotated 180 deg) -> stacked dataset
  2. Create mirrored copy from the stacked dataset (horizontal flip videos,
     swap left/right arms, swap left_wrist/right_wrist, wrist rotated 180)
  3. Merge original stacked + mirrored -> final dataset (2x episodes)

Mirroring logic reference:
  D:/VLA/kai0/train_deploy_alignment/data_augment/space_mirroring.py

Or run the two steps separately:
  python stack_front_cameras.py --src-path <src> --tgt-path <stacked>
  python space_mirroring.py full --src-path <stacked> --mirror-path <mirror> --merge-path <final> --repo-id xxx
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

LEFT_ARM_DIM = 7
RIGHT_ARM_DIM = 7
TOTAL_ARM_DIM = LEFT_ARM_DIM + RIGHT_ARM_DIM  # 14


# ==================== Video helpers ====================

def _get_episodes(chunk_dir: Path) -> list[Path]:
    for cam_dir in sorted(chunk_dir.iterdir()):
        if cam_dir.is_dir():
            eps = sorted(f for f in cam_dir.iterdir() if f.suffix == ".mp4")
            if eps:
                return eps
    return []


def stack_front_videos(
    front_path: str, front1_path: str, output_path: str
) -> tuple[int, int]:
    """Stack front (top) + front_1 rotated 180 (bottom). Returns (new_h, new_w)."""
    cap_top = cv2.VideoCapture(front_path)
    cap_bottom = cv2.VideoCapture(front1_path)
    if not cap_top.isOpened():
        raise RuntimeError(f"Cannot open: {front_path}")
    if not cap_bottom.isOpened():
        raise RuntimeError(f"Cannot open: {front1_path}")

    fps = int(cap_top.get(cv2.CAP_PROP_FPS))
    w = int(cap_top.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_top = int(cap_top.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_bottom = int(cap_bottom.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w != int(cap_bottom.get(cv2.CAP_PROP_FRAME_WIDTH)):
        cap_top.release()
        cap_bottom.release()
        raise RuntimeError("front and front_1 width mismatch")

    new_h = h_top + h_bottom

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, new_h))

    while True:
        ret_top, frame_top = cap_top.read()
        ret_bottom, frame_bottom = cap_bottom.read()
        if not ret_top or not ret_bottom:
            break
        frame_bottom = cv2.rotate(frame_bottom, cv2.ROTATE_180)
        out.write(np.vstack([frame_top, frame_bottom]))

    cap_top.release()
    cap_bottom.release()
    out.release()
    return new_h, w


def flip_video_h(input_path: str, output_path: str) -> None:
    """Horizontally mirror-flip all frames."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.flip(frame, 1))

    cap.release()
    out.release()


def flip_h_and_rotate_180(input_path: str, output_path: str) -> None:
    """Horizontal mirror-flip then rotate 180 degrees for wrist cameras."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        out.write(frame)

    cap.release()
    out.release()


# ==================== Array helpers ====================

def swap_left_right_arms(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 0:
        return arr
    flat = arr.flatten()
    if len(flat) != TOTAL_ARM_DIM:
        raise ValueError(f"Expected {TOTAL_ARM_DIM} dims, got {len(flat)}")
    swapped = np.concatenate([flat[LEFT_ARM_DIM:], flat[:LEFT_ARM_DIM]])
    if arr.ndim > 1:
        swapped = swapped.reshape(arr.shape)
    return swapped


def swap_list_left_right(arr: list) -> list:
    if len(arr) < TOTAL_ARM_DIM:
        raise ValueError(f"Expected at least {TOTAL_ARM_DIM}, got {len(arr)}")
    swapped = arr[LEFT_ARM_DIM:TOTAL_ARM_DIM] + arr[:LEFT_ARM_DIM]
    if len(arr) > TOTAL_ARM_DIM:
        swapped = swapped + arr[TOTAL_ARM_DIM:]
    return swapped


# ==================== Metadata helpers ====================

def build_merged_info(stacked_info: dict) -> dict:
    """Build info.json for the merged (stacked + mirrored) dataset."""
    info = json.loads(json.dumps(stacked_info))

    # Doubled counts
    info["total_episodes"] = info["total_episodes"] * 2
    info["total_frames"] = info["total_frames"] * 2

    if "splits" in info and "train" in info["splits"]:
        info["splits"]["train"] = f"0:{info['total_episodes']}"

    return info


def build_merged_episodes_jsonl(stacked_meta: Path, tgt_meta: Path) -> None:
    """Double episodes: original at i*2, mirrored at i*2+1."""
    src_file = stacked_meta / "episodes.jsonl"
    if not src_file.exists():
        return
    with open(src_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    tgt_meta.mkdir(parents=True, exist_ok=True)
    with open(tgt_meta / "episodes.jsonl", "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            ep = json.loads(line)
            ep["episode_index"] = i * 2
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            ep_mir = dict(ep)
            ep_mir["episode_index"] = i * 2 + 1
            f.write(json.dumps(ep_mir, ensure_ascii=False) + "\n")


def build_merged_stats_jsonl(stacked_meta: Path, tgt_meta: Path) -> None:
    """Double stats: original at i*2, mirrored (arms swapped) at i*2+1."""
    src_file = stacked_meta / "episodes_stats.jsonl"
    if not src_file.exists():
        return
    with open(src_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    tgt_meta.mkdir(parents=True, exist_ok=True)
    with open(tgt_meta / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            data = json.loads(line)
            data["episode_index"] = i * 2
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            # Mirrored: swap arm stats
            mir = json.loads(json.dumps(data))
            mir["episode_index"] = i * 2 + 1
            if "stats" in mir:
                for key in ("observation.state", "action"):
                    if key in mir["stats"]:
                        item = mir["stats"][key]
                        for metric in ("mean", "std", "min", "max"):
                            if metric in item:
                                item[metric] = swap_list_left_right(item[metric])
            f.write(json.dumps(mir, ensure_ascii=False) + "\n")


# ==================== Main pipeline ====================

def augment_dataset(src_path: str, tgt_path: str) -> None:
    """
    Full pipeline:
    Step 1: Stack front + front_1 (front_1 rotated 180) in place
    Step 2: Create mirrored episodes
    Step 3: Merge into final dataset
    """
    src_root = Path(src_path)
    tgt_root = Path(tgt_path)

    if not src_root.exists():
        raise RuntimeError(f"Source does not exist: {src_root}")

    with open(src_root / "meta" / "info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    num_src_episodes = info["total_episodes"]
    stacked_h: int | None = None
    stacked_w: int | None = None

    print("=" * 60)
    print("Full Augmentation Pipeline")
    print(f"  Source: {src_root}")
    print(f"  Target: {tgt_root}")
    print(f"  Source episodes: {num_src_episodes}")
    print("=" * 60)

    src_videos = src_root / "videos"
    tgt_videos = tgt_root / "videos"

    # ---- Step 1: Stack front + front_1 for ALL output episodes ----
    print("\n[1/5] Stacking front + front_1 videos (front_1 rotated 180)...")
    if not src_videos.exists():
        raise RuntimeError("No videos directory")

    for chunk_dir in sorted(src_videos.iterdir()):
        if not chunk_dir.is_dir():
            continue
        print(f"  Chunk: {chunk_dir.name}")
        episodes = _get_episodes(chunk_dir)
        if not episodes:
            continue

        for ep_path in tqdm(episodes, desc="  Episodes"):
            ep_name = ep_path.stem
            front_src = str(chunk_dir / "observation.images.front" / f"{ep_name}.mp4")
            front1_src = str(chunk_dir / "observation.images.front_1" / f"{ep_name}.mp4")

            # --- a) Original (stacked only): episode i*2 ---
            orig_ep = f"episode_{int(ep_name.split('_')[-1]) * 2:06d}"
            tgt_front_dir = tgt_videos / chunk_dir.name / "observation.images.front"
            orig_front = str(tgt_front_dir / f"{orig_ep}.mp4")
            h, w = stack_front_videos(front_src, front1_src, orig_front)
            if stacked_h is None:
                stacked_h, stacked_w = h, w

            # Copy wrist videos for original
            for wrist_view in (
                "observation.images.left_wrist",
                "observation.images.right_wrist",
            ):
                src_wrist = chunk_dir / wrist_view / f"{ep_name}.mp4"
                if src_wrist.exists():
                    tgt_dir = tgt_videos / chunk_dir.name / wrist_view
                    tgt_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src_wrist), str(tgt_dir / f"{orig_ep}.mp4"))

            # --- b) Mirrored (stacked + flip + swap): episode i*2 + 1 ---
            mir_ep = f"episode_{int(ep_name.split('_')[-1]) * 2 + 1:06d}"
            # Stack first, then mirror-flip
            tmp_stacked = str(tgt_front_dir / f"_tmp_{mir_ep}.mp4")
            stack_front_videos(front_src, front1_src, tmp_stacked)
            mir_front = str(tgt_front_dir / f"{mir_ep}.mp4")
            flip_video_h(tmp_stacked, mir_front)
            os.remove(tmp_stacked)

            # Wrist: horizontal flip + rotate 180 + swap left <-> right
            for wrist_view in (
                "observation.images.left_wrist",
                "observation.images.right_wrist",
            ):
                src_wrist = chunk_dir / wrist_view / f"{ep_name}.mp4"
                if not src_wrist.exists():
                    continue
                if "left_wrist" in wrist_view:
                    out_view = "observation.images.right_wrist"
                else:
                    out_view = "observation.images.left_wrist"
                tgt_dir = tgt_videos / chunk_dir.name / out_view
                tgt_dir.mkdir(parents=True, exist_ok=True)
                flip_h_and_rotate_180(str(src_wrist), str(tgt_dir / f"{mir_ep}.mp4"))

    # ---- Step 2: Process parquet data (original + mirrored) ----
    print("\n[2/5] Processing parquet data (original + mirrored)...")
    src_data = src_root / "data"
    tgt_data = tgt_root / "data"

    if not src_data.exists():
        raise RuntimeError("No data directory")

    for chunk_dir in sorted(src_data.iterdir()):
        if not chunk_dir.is_dir():
            continue
        tgt_chunk = tgt_data / chunk_dir.name
        tgt_chunk.mkdir(parents=True, exist_ok=True)

        for parquet_file in sorted(chunk_dir.iterdir()):
            if parquet_file.suffix != ".parquet":
                continue

            df = pd.read_parquet(str(parquet_file))

            # Original episode (unchanged)
            orig_df = df.copy()
            orig_df["episode_index"] = orig_df["episode_index"] * 2

            # Mirrored episode (swap arms)
            mir_df = df.copy()
            mir_df["episode_index"] = mir_df["episode_index"] * 2 + 1
            for col in ("observation.state", "action"):
                if col in mir_df.columns:
                    mir_df[col] = mir_df[col].apply(swap_left_right_arms)

            merged = pd.concat([orig_df, mir_df], ignore_index=True)
            orig_ep_idx = int(parquet_file.stem.split("_")[-1]) * 2
            merged.to_parquet(
                str(tgt_chunk / f"episode_{orig_ep_idx:06d}.parquet"), index=False
            )

    print("  Done.")

    # ---- Step 3: Update info.json ----
    print("\n[3/5] Updating info.json...")
    stacked_info = json.loads(json.dumps(info))

    # Remove front_1, update front dims
    features = stacked_info["features"]
    if "observation.images.front" in features:
        features["observation.images.front"]["shape"] = [stacked_h, stacked_w, 3]
        features["observation.images.front"]["names"] = ["height", "width", "channels"]
        if "info" in features["observation.images.front"]:
            features["observation.images.front"]["info"]["video.height"] = stacked_h
            features["observation.images.front"]["info"]["video.width"] = stacked_w
    features.pop("observation.images.front_1", None)
    stacked_info["total_videos"] = 3

    tgt_meta = tgt_root / "meta"
    tgt_meta.mkdir(parents=True, exist_ok=True)
    with open(tgt_meta / "info.json", "w", encoding="utf-8") as f:
        json.dump(build_merged_info(stacked_info), f, indent=2, ensure_ascii=False)
    print("  [OK]")

    # ---- Step 4: Build episodes metadata ----
    print("\n[4/5] Generating episodes metadata...")
    # Write stacked info first (needed as base for merged builders)
    src_meta = src_root / "meta"
    build_merged_episodes_jsonl(src_meta, tgt_meta)
    print("  [OK] episodes.jsonl")
    build_merged_stats_jsonl(src_meta, tgt_meta)
    print("  [OK] episodes_stats.jsonl")

    # tasks.jsonl
    tasks_src = src_meta / "tasks.jsonl"
    if tasks_src.exists():
        shutil.copy2(str(tasks_src), str(tgt_meta / "tasks.jsonl"))
        print("  [OK] tasks.jsonl")

    # ---- Step 5: Copy auxiliary files ----
    print("\n[5/5] Copying auxiliary files...")
    writer_log = src_root / "writer.log"
    if writer_log.exists():
        shutil.copy2(str(writer_log), str(tgt_root / "writer.log"))
        print("  [OK] writer.log")

    print("\n" + "=" * 60)
    print(f"Pipeline complete: {num_src_episodes} -> {num_src_episodes * 2} episodes")
    print(f"  - Episodes 0,2,4,... : stacked (front + front_1 rotated 180), no mirror")
    print(f"  - Episodes 1,3,5,... : stacked + horizontal flip + arm swap + wrist swap + wrist rotate 180")
    print(f"  - Front dimensions: {stacked_h}x{stacked_w}")
    print(f"Output: {tgt_root}")
    print("=" * 60)


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="Full augmentation: stack front cameras + mirror doubling"
    )
    parser.add_argument("--src-path", required=True, help="Source dataset path")
    parser.add_argument("--tgt-path", required=True, help="Output dataset path")
    args = parser.parse_args()

    augment_dataset(args.src_path, args.tgt_path)


if __name__ == "__main__":
    main()
