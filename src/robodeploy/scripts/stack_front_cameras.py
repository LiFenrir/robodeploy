#!/usr/bin/env python3
"""
Pre-process bi_s1 LeRobot dataset: vertically stack observation.images.front
and observation.images.front_1 into a single front view.

- front_1 is rotated 180 degrees around center before stacking below front (skip with --no-rotate).
- Output dataset has 3 video views (front, left_wrist, right_wrist) instead of 4.
- Front dimensions become 960x848 (was two 480x848).
- All other data (parquet, wrist videos, meta) is copied as-is.

After this, use space_mirroring.py for mirror augmentation and merging:
  python space_mirroring.py full --src-path <stacked> --mirror-path <mirror> --merge-path <final> --repo-id xxx

Reference: D:/datasets/bi_s1_0521
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


def _probe_resolution(video_path: str) -> tuple[int, int, float]:
    """Get (width, height, fps) from a video via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    w, h, fps_str = result.stdout.strip().split(",")
    num, den = fps_str.split("/")
    return int(w), int(h), float(num) / float(den)


def _get_ffmpeg_video_encode_args() -> list[str]:
    """Return ffmpeg encoder arguments, probing for libsvtav1 availability.

    Uses libsvtav1 (AV1) if available, falling back to libx264 (H.264).
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True,
        )
        if "libsvtav1" in result.stdout:
            return ["-c:v", "libsvtav1", "-crf", "30"]
    except Exception:
        pass
    # Fallback to widely-available H.264
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]


def stack_front_videos(
    front_path: str,
    front1_path: str,
    output_path: str,
    rotate: bool = True,
) -> tuple[int, int]:
    """Stack front (top) + front_1 (bottom) via ffmpeg. Returns (new_h, new_w)."""
    w1, h1, fps = _probe_resolution(front_path)
    w2, h2, _ = _probe_resolution(front1_path)
    if w1 != w2:
        raise RuntimeError(f"front ({w1}x{h1}) and front_1 ({w2}x{h2}) width mismatch")

    new_h = h1 + h2
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if rotate:
        filter_str = "[1:v]rotate=PI:ow=iw:oh=ih[v1]; [0:v][v1]vstack=inputs=2"
    else:
        filter_str = "[0:v][1:v]vstack=inputs=2"

    cmd = [
        "ffmpeg", "-y",
        "-i", front_path,
        "-i", front1_path,
        "-filter_complex",
        filter_str,
        *_get_ffmpeg_video_encode_args(),
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

    return new_h, w1


def _get_episodes(chunk_dir: Path) -> list[Path]:
    for cam_dir in sorted(chunk_dir.iterdir()):
        if cam_dir.is_dir():
            eps = sorted(f for f in cam_dir.iterdir() if f.suffix == ".mp4")
            if eps:
                return eps
    return []


def update_info_json(info: dict, stacked_h: int, stacked_w: int) -> dict:
    """Update info.json for the stacked dataset."""
    features = info["features"]

    # Update front dims
    if "observation.images.front" in features:
        features["observation.images.front"]["shape"] = [stacked_h, stacked_w, 3]
        features["observation.images.front"]["names"] = ["height", "width", "channels"]
        if "info" in features["observation.images.front"]:
            features["observation.images.front"]["info"]["video.height"] = stacked_h
            features["observation.images.front"]["info"]["video.width"] = stacked_w

    # Remove front_1
    features.pop("observation.images.front_1", None)

    # 3 video views now
    info["total_videos"] = 3

    return info


def process_dataset(src_path: str, tgt_path: str, rotate: bool = True) -> None:
    src_root = Path(src_path)
    tgt_root = Path(tgt_path)

    if not src_root.exists():
        raise RuntimeError(f"Source does not exist: {src_root}")

    with open(src_root / "meta" / "info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    stacked_h: int | None = None
    stacked_w: int | None = None

    rot_msg = "rotated 180" if rotate else "no rotation"
    print("=" * 50)
    print(f"Front Camera Stacking (front + front_1, {rot_msg})")
    print(f"  Source: {src_root}")
    print(f"  Target: {tgt_root}")
    print("=" * 50)

    # --- Videos ---
    print("\n[1/3] Stacking front + front_1 videos...")
    src_videos = src_root / "videos"
    tgt_videos = tgt_root / "videos"

    if src_videos.exists():
        for chunk_dir in sorted(src_videos.iterdir()):
            if not chunk_dir.is_dir():
                continue
            print(f"  Chunk: {chunk_dir.name}")
            episodes = _get_episodes(chunk_dir)
            if not episodes:
                continue

            for ep_path in tqdm(episodes, desc=f"  Episodes"):
                ep_name = ep_path.stem
                front_src = str(chunk_dir / "observation.images.front" / f"{ep_name}.mp4")
                front1_src = str(chunk_dir / "observation.images.front_1" / f"{ep_name}.mp4")

                # Stack front + front_1
                tgt_front_dir = tgt_videos / chunk_dir.name / "observation.images.front"
                tgt_front = str(tgt_front_dir / f"{ep_name}.mp4")
                h, w = stack_front_videos(front_src, front1_src, tgt_front, rotate=rotate)
                if stacked_h is None:
                    stacked_h, stacked_w = h, w

                # Copy wrist videos as-is
                for wrist_view in (
                    "observation.images.left_wrist",
                    "observation.images.right_wrist",
                ):
                    src_vid = chunk_dir / wrist_view / f"{ep_name}.mp4"
                    if src_vid.exists():
                        tgt_dir = tgt_videos / chunk_dir.name / wrist_view
                        tgt_vid = tgt_dir / f"{ep_name}.mp4"
                        tgt_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(src_vid), str(tgt_vid))

    # --- Data (copy as-is) ---
    print("\n[2/3] Copying parquet data...")
    src_data = src_root / "data"
    tgt_data = tgt_root / "data"
    if src_data.exists():
        shutil.copytree(str(src_data), str(tgt_data), dirs_exist_ok=True)

    # --- Metadata ---
    print("\n[3/3] Updating metadata...")
    tgt_meta = tgt_root / "meta"
    tgt_meta.mkdir(parents=True, exist_ok=True)

    # info.json
    out_info = update_info_json(info, stacked_h, stacked_w)
    with open(tgt_meta / "info.json", "w", encoding="utf-8") as f:
        json.dump(out_info, f, indent=2, ensure_ascii=False)
    print("  [OK] info.json")

    # Other meta files - copy as-is
    for meta_file in ["episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"]:
        src_file = src_root / "meta" / meta_file
        if src_file.exists():
            shutil.copy2(str(src_file), str(tgt_meta / meta_file))
            print(f"  [OK] {meta_file}")

    # writer.log
    writer_log = src_root / "writer.log"
    if writer_log.exists():
        shutil.copy2(str(writer_log), str(tgt_root / "writer.log"))

    print("\n" + "=" * 50)
    print(f"Done. Front: {stacked_h}x{stacked_w} (was 480x848 + 480x848)")
    print(f"Output: {tgt_root}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Stack front + front_1 cameras (front_1 rotated 180) for bi_s1 dataset"
    )
    parser.add_argument("--src-path", required=True, help="Source dataset path")
    parser.add_argument("--tgt-path", required=True, help="Output dataset path")
    parser.add_argument("--no-rotate", action="store_true", help="Skip 180-degree rotation of front_1")
    args = parser.parse_args()

    process_dataset(args.src_path, args.tgt_path, rotate=not args.no_rotate)


if __name__ == "__main__":
    main()