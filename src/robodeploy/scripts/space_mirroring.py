#!/usr/bin/env python3
"""
space_mirror.py

Space Mirror core functionality: dual-arm data mirroring and data augmentation
- Swap left/right arm data (parquet, json, jsonl)
- Flip videos (horizontal mirroring)
- Merge original and mirrored datasets
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm

_utils_dir = Path(__file__).parent
_MERGE_SCRIPT = _utils_dir / "merge_lerobot_datasets.py"
MERGE_AVAILABLE = _MERGE_SCRIPT.exists()


# ==================== Core Utility Functions ====================

def swap_arms_in_array(arr: np.ndarray, left_dim: int = 7, right_dim: int = 7) -> np.ndarray:
    """Swap the first left_dim dimensions with the last right_dim dimensions of an array"""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    if arr.ndim == 0:
        return arr
    
    arr_flat = arr.flatten()
    total_dim = left_dim + right_dim
    
    if len(arr_flat) != total_dim:
        raise ValueError(
            f"Array dimension mismatch: expected {total_dim} dims (left{left_dim} + right{right_dim}), "
            f"got {len(arr_flat)} dims"
        )
    
    left_arm = arr_flat[:left_dim].copy()
    right_arm = arr_flat[left_dim:left_dim + right_dim].copy()
    swapped = np.concatenate([right_arm, left_arm])

    # Negate j1(0), j5(4), j6(5) in both halves after swap
    negate_indices = [0, 4, 5, left_dim, left_dim + 4, left_dim + 5]
    swapped[negate_indices] *= -1

    if arr.ndim > 1:
        swapped = swapped.reshape(arr.shape)

    return swapped


def swap_array_dims_list(arr: List[float], left_dim: int = 7, right_dim: int = 7, keep_padding: bool = True) -> List[float]:
    """Swap the first and last dimensions of a list (for JSON/JSONL)"""
    if not isinstance(arr, list):
        arr = list(arr)
    
    total_dim = left_dim + right_dim
    
    if len(arr) < total_dim:
        raise ValueError(
            f"Insufficient array dimensions: expected at least {total_dim} dims (left{left_dim} + right{right_dim}), "
            f"got {len(arr)} dims"
        )
    
    left_arm = arr[:left_dim].copy()
    right_arm = arr[left_dim:left_dim + right_dim].copy()
    swapped = right_arm + left_arm

    # Negate j1(0), j5(4), j6(5) in both halves after swap
    negate_indices = [0, 4, 5, left_dim, left_dim + 4, left_dim + 5]
    for i in negate_indices:
        if i < len(swapped):
            swapped[i] = -swapped[i]

    if keep_padding and len(arr) > total_dim:
        padding = arr[total_dim:]
        swapped = swapped + padding
    
    return swapped


# ==================== Parquet Processing ====================

def swap_arms_in_parquet(
    input_path: Path,
    output_path: Path,
    columns: Optional[List[str]] = None,
    left_dim: int = 7,
    right_dim: int = 7,
) -> Tuple[str, bool, str]:
    """Process a single parquet file, swap the first and last dimensions of specified columns"""
    try:
        df = pd.read_parquet(str(input_path))
        
        if columns is None:
            columns_to_process = []
            for col in ['observation.state', 'action']:
                if col in df.columns:
                    columns_to_process.append(col)
        else:
            columns_to_process = [col for col in columns if col in df.columns]
        
        if not columns_to_process:
            return (str(input_path), False, "No columns found to process")
        
        for col in columns_to_process:
            if col not in df.columns:
                continue
            
            if df[col].dtype != object:
                return (
                    str(input_path),
                    False,
                    f"Column {col} is not object type (not nested array), skipping"
                )
            
            swapped_values = []
            for idx, val in enumerate(df[col]):
                try:
                    if isinstance(val, (list, tuple)):
                        arr = np.array(val)
                    elif isinstance(val, np.ndarray):
                        arr = val.copy()
                    else:
                        return (
                            str(input_path),
                            False,
                            f"Unsupported data type for column {col} row {idx}: {type(val)}"
                        )
                    
                    swapped_arr = swap_arms_in_array(arr, left_dim, right_dim)
                    swapped_values.append(swapped_arr)
                
                except Exception as e:
                    return (
                        str(input_path),
                        False,
                        f"Error processing column {col} row {idx}: {str(e)}"
                    )
            
            df[col] = swapped_values
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(output_path), index=False)
        
        return (
            str(input_path),
            True,
            f"Successfully processed {len(columns_to_process)} columns: {', '.join(columns_to_process)}"
        )
    
    except Exception as e:
        return (str(input_path), False, f"Error: {str(e)}")


def process_parquet_files(
    input_dir: Path,
    output_dir: Path,
    columns: Optional[List[str]] = None,
    left_dim: int = 7,
    right_dim: int = 7,
    num_workers: int = 4,
) -> None:
    """Batch process parquet files"""
    parquet_files = list(input_dir.rglob('*.parquet'))
    
    if not parquet_files:
        print(f"Warning: No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    
    def get_output_path(input_file: Path) -> Path:
        relative = input_file.relative_to(input_dir)
        return output_dir / relative
    
    tasks = [(f, get_output_path(f)) for f in parquet_files]
    
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(swap_arms_in_parquet, inp, out, columns, left_dim, right_dim): inp
            for inp, out in tasks
        }
        
        for future in tqdm(as_completed(future_to_file), total=len(tasks), desc="Processing parquet"):
            input_path = future_to_file[future]
            try:
                result_path, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    print(f"✗ [{result_path}] {message}")
                    fail_count += 1
            except Exception as e:
                print(f"✗ [{input_path}] Processing exception: {str(e)}")
                fail_count += 1
    
    print(f"Parquet processing complete: {success_count} succeeded, {fail_count} failed")


# ==================== JSON Processing ====================

def process_norm_stats_json(
    input_path: Path,
    output_path: Path,
    left_dim: int = 7,
    right_dim: int = 7,
) -> Tuple[str, bool, str]:
    """Process norm_stats.json file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "norm_stats" not in data:
            return (str(input_path), False, "Field 'norm_stats' not found in JSON file")
        
        norm_stats = data["norm_stats"]
        
        for key in ["state", "actions"]:
            if key not in norm_stats:
                continue
            
            stat_item = norm_stats[key]
            
            for stat_key in ["mean", "std", "q01", "q99"]:
                if stat_key in stat_item:
                    try:
                        stat_item[stat_key] = swap_array_dims_list(
                            stat_item[stat_key],
                            left_dim,
                            right_dim,
                            keep_padding=True
                        )
                    except Exception as e:
                        print(f"Warning: Error processing norm_stats.{key}.{stat_key}: {e}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return (str(input_path), True, "Processing successful")
    
    except Exception as e:
        return (str(input_path), False, f"Error: {str(e)}")


# ==================== JSONL Processing ====================

def swap_stats_dims(
    stats_dict: Dict[str, Any],
    left_dim: int = 7,
    right_dim: int = 7
) -> Dict[str, Any]:
    """Swap left/right arm data in stats dictionary"""
    # Swap hand_left and hand_right
    if "observation.images.hand_left" in stats_dict and "observation.images.hand_right" in stats_dict:
        hand_left = stats_dict["observation.images.hand_left"]
        hand_right = stats_dict["observation.images.hand_right"]
        stats_dict["observation.images.hand_left"] = hand_right
        stats_dict["observation.images.hand_right"] = hand_left
    
    # Process observation.state and action
    for key in ["observation.state", "action"]:
        if key not in stats_dict:
            continue
        
        stat_item = stats_dict[key]
        
        for stat_key in ["min", "max", "mean", "std"]:
            if stat_key in stat_item:
                try:
                    stat_item[stat_key] = swap_array_dims_list(
                        stat_item[stat_key],
                        left_dim,
                        right_dim
                    )
                except Exception as e:
                    print(f"Warning: Error processing {key}.{stat_key}: {e}")
    
    return stats_dict


def process_episodes_stats_jsonl(
    input_path: Path,
    output_path: Path,
    left_dim: int = 7,
    right_dim: int = 7,
) -> Tuple[str, bool, str]:
    """Process episodes_stats.jsonl file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        processed_lines = []
        processed_count = 0
        error_count = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                processed_lines.append('')
                continue
            
            try:
                data = json.loads(line)
                
                if "stats" in data and isinstance(data["stats"], dict):
                    data["stats"] = swap_stats_dims(data["stats"], left_dim, right_dim)
                    processed_count += 1
                
                processed_lines.append(json.dumps(data, ensure_ascii=False))
            
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"Error: JSON parsing failed at line {line_num}: {e}")
                processed_lines.append(line)
            except Exception as e:
                error_count += 1
                print(f"Error: Processing failed at line {line_num}: {e}")
                processed_lines.append(line)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
            if processed_lines and processed_lines[-1]:
                f.write('\n')
        
        message = f"Successfully processed {processed_count} entries"
        if error_count > 0:
            message += f", {error_count} errors"
        
        return (str(input_path), True, message)
    
    except Exception as e:
        return (str(input_path), False, f"Error: {str(e)}")


# ==================== Video Processing ====================

def flip_video(input_path: str, output_path: str, transform: str = "hflip") -> Tuple[str, bool, str]:
    """Process a single video file via ffmpeg, output AV1 (libsvtav1)."""

    transform: "copy" = no filter, "hflip" = horizontal mirror,
               "hflip+rotate180" = hflip then 180-degree rotation.
    """
    vf_map = {
        "copy": "null",
        "hflip": "hflip",
        "hflip+rotate180": "hflip,transpose=2,transpose=2",
    }
    vf = vf_map.get(transform, "hflip")
    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libsvtav1", "-crf", "30",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return (input_path, False, f"ffmpeg failed: {result.stderr[-200:]}")
        return (input_path, True, "ok")
    except Exception as e:
        return (input_path, False, f"Error: {str(e)}")


def process_videos(
    input_dir: Path,
    output_dir: Path,
    num_workers: int = 4,
    transform: str = "hflip",
) -> None:
    """Batch process video files"""
    video_files = list(input_dir.rglob('*.mp4'))

    if not video_files:
        print(f"Warning: No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} video files")

    def get_output_path(input_file: Path) -> Path:
        relative = input_file.relative_to(input_dir)
        return output_dir / relative

    tasks = [(str(f), str(get_output_path(f)), transform) for f in video_files]

    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_video = {
            executor.submit(flip_video, inp, out, tr): inp
            for inp, out, tr in tasks
        }
        
        for future in tqdm(as_completed(future_to_video), total=len(tasks), desc="Processing videos"):
            input_path = future_to_video[future]
            try:
                result_path, success, message = future.result()
                if success:
                    success_count += 1
                else:
                    print(f"✗ [{result_path}] {message}")
                    fail_count += 1
            except Exception as e:
                print(f"✗ [{input_path}] Processing exception: {str(e)}")
                fail_count += 1
    
    print(f"Video processing complete: {success_count} succeeded, {fail_count} failed")


# ==================== Dataset Merging ====================

def merge_lerobot_datasets(
    src_paths: List[str],
    tgt_path: str,
    repo_id: str,
    fps: int = 30,
    robot_type: str = "agilex",
    features: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> None:
    """Merge multiple LeRobot datasets by calling merge_lerobot_datasets.py."""
    if not MERGE_AVAILABLE:
        raise RuntimeError(f"Merge script not found: {_MERGE_SCRIPT}")
    cmd = [
        sys.executable, str(_MERGE_SCRIPT),
        "--datasets", *src_paths,
        "--output_dir", tgt_path,
        "--repo_id", repo_id,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Merge failed with exit code {result.returncode}")


# ==================== Main Functions ====================

def create_mirror_dataset(
    src_path: str,
    tgt_path: str,
    left_dim: int = 7,
    right_dim: int = 7,
    num_workers: int = 4,
    flip_views: list[str] | None = None,
    swap_left_view: str = "observation.images.hand_left",
    swap_right_view: str = "observation.images.hand_right",
) -> None:
    """Create mirrored dataset"""
    if flip_views is None:
        flip_views = ["observation.images.top_head"]
    src_root = Path(src_path).expanduser().resolve()
    tgt_root = Path(tgt_path).expanduser().resolve()
    
    if not src_root.exists():
        raise RuntimeError(f"Source path does not exist: {src_root}")
    
    print("=" * 60)
    print("Starting to create mirrored dataset")
    print("=" * 60)
    print(f"Source path: {src_root}")
    print(f"Target path: {tgt_root}")
    print()
    
    # 1. Process norm_stats.json
    print("[1/5] Processing norm_stats.json...")
    src_norm_stats = src_root / "norm_stats.json"
    if src_norm_stats.exists():
        tgt_norm_stats = tgt_root / "norm_stats.json"
        result, success, msg = process_norm_stats_json(src_norm_stats, tgt_norm_stats, left_dim, right_dim)
        if success:
            print(f"✓ norm_stats.json processing complete")
        else:
            print(f"✗ norm_stats.json processing failed: {msg}")
    else:
        print("  Skipping: norm_stats.json does not exist")
    print()
    
    # 2. Process episodes_stats.jsonl
    print("[2/5] Processing episodes_stats.jsonl...")
    src_episodes_stats = src_root / "meta" / "episodes_stats.jsonl"
    if src_episodes_stats.exists():
        tgt_episodes_stats = tgt_root / "meta" / "episodes_stats.jsonl"
        result, success, msg = process_episodes_stats_jsonl(src_episodes_stats, tgt_episodes_stats, left_dim, right_dim)
        if success:
            print(f"✓ episodes_stats.jsonl processing complete: {msg}")
        else:
            print(f"✗ episodes_stats.jsonl processing failed: {msg}")
    else:
        print("  Skipping: episodes_stats.jsonl does not exist")
    print()
    
    # 3. Process parquet files
    print("[3/5] Processing parquet files...")
    src_data_dir = src_root / "data"
    if src_data_dir.exists():
        tgt_data_dir = tgt_root / "data"
        process_parquet_files(src_data_dir, tgt_data_dir, None, left_dim, right_dim, num_workers)
        print("✓ All parquet files processing complete")
    else:
        print("  Skipping: data directory does not exist")
    print()
    
    # 4. Process video files
    print("[4/5] Flipping video files...")
    src_videos_dir = src_root / "videos"
    if src_videos_dir.exists():
        # Find all chunk directories
        chunks = sorted([d for d in src_videos_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")])
        
        if not chunks:
            print("  Warning: No chunk directories found")
        else:
            print(f"  Found {len(chunks)} chunk directories")
            
            for chunk_dir in chunks:
                chunk_name = chunk_dir.name
                tgt_chunk_dir = tgt_root / "videos" / chunk_name
                
                # Direct views (front etc.) — horizontal flip
                for view in flip_views:
                    src_view_dir = chunk_dir / view
                    if src_view_dir.exists() and src_view_dir.is_dir():
                        tgt_view_dir = tgt_chunk_dir / view
                        print(f"    Processing {chunk_name}/{view} (hflip)...")
                        process_videos(src_view_dir, tgt_view_dir, num_workers, transform="hflip")

                # Swap left/right views: hflip + rotate180
                right_src = chunk_dir / swap_right_view
                if right_src.exists() and right_src.is_dir():
                    left_tgt = tgt_chunk_dir / swap_left_view
                    print(f"    Processing {chunk_name}/{swap_right_view} -> {swap_left_view} (hflip+rotate180)...")
                    process_videos(right_src, left_tgt, num_workers, transform="hflip+rotate180")

                left_src = chunk_dir / swap_left_view
                if left_src.exists() and left_src.is_dir():
                    right_tgt = tgt_chunk_dir / swap_right_view
                    print(f"    Processing {chunk_name}/{swap_left_view} -> {swap_right_view} (hflip+rotate180)...")
                    process_videos(left_src, right_tgt, num_workers, transform="hflip+rotate180")
        
        print("✓ All video files processing complete")
    else:
        print("  Skipping: videos directory does not exist")
    print()
    
    # 5. Copy other meta files
    print("[5/5] Copying other meta files...")
    src_meta_dir = src_root / "meta"
    if src_meta_dir.exists():
        tgt_meta_dir = tgt_root / "meta"
        tgt_meta_dir.mkdir(parents=True, exist_ok=True)
        
        for meta_file in ["episodes.jsonl", "info.json", "tasks.jsonl"]:
            src_file = src_meta_dir / meta_file
            if src_file.exists():
                shutil.copy2(src_file, tgt_meta_dir / meta_file)
                print(f"  ✓ Copied {meta_file}")
    
    print("✓ Mirrored dataset creation complete")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Space Mirror: Dual-arm data mirroring and data augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create mirrored dataset
  python space_mirror.py create-mirror --src-path /path/to/original --tgt-path /path/to/mirror
  
  # Merge original and mirrored datasets
  python space_mirror.py merge --src-paths /path/to/original /path/to/mirror --tgt-path /path/to/merged --repo-id my_dataset
  
  # Full pipeline (create mirror and merge)
  python space_mirror.py full --src-path /path/to/original --mirror-path /path/to/mirror --merge-path /path/to/merged --repo-id my_dataset
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # create-mirror command
    parser_create = subparsers.add_parser('create-mirror', help='Create mirrored dataset')
    parser_create.add_argument('--src-path', required=True, help='Source dataset path')
    parser_create.add_argument('--tgt-path', required=True, help='Target mirrored dataset path')
    parser_create.add_argument('--left-dim', type=int, default=7, help='Left arm dimension (default: 7)')
    parser_create.add_argument('--right-dim', type=int, default=7, help='Right arm dimension (default: 7)')
    parser_create.add_argument('--num-workers', type=int, default=4, help='Number of parallel worker processes (default: 4)')
    parser_create.add_argument('--flip-views', nargs='+', default=None,
                               help='View names to flip directly (default: observation.images.top_head)')
    parser_create.add_argument('--swap-left-view', type=str, default='observation.images.hand_left',
                               help='Left view to swap with right (default: observation.images.hand_left)')
    parser_create.add_argument('--swap-right-view', type=str, default='observation.images.hand_right',
                               help='Right view to swap with left (default: observation.images.hand_right)')
    
    # merge command
    parser_merge = subparsers.add_parser('merge', help='Merge datasets')
    parser_merge.add_argument('--src-paths', nargs='+', required=True, help='Source dataset paths list')
    parser_merge.add_argument('--tgt-path', required=True, help='Target merged dataset path')
    parser_merge.add_argument('--repo-id', required=True, help='Dataset repo_id')
    parser_merge.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')
    parser_merge.add_argument('--robot-type', type=str, default='agilex', help='Robot type (default: agilex)')
    parser_merge.add_argument('--features-json', type=str, default=None, help='Path to features.json file')
    parser_merge.add_argument('--force', action='store_true', help='Force merge (ignore conflicts)')
    
    # full command
    parser_full = subparsers.add_parser('full', help='Full pipeline: create mirror and merge')
    parser_full.add_argument('--src-path', required=True, help='Source dataset path')
    parser_full.add_argument('--mirror-path', required=True, help='Mirrored dataset path')
    parser_full.add_argument('--merge-path', required=True, help='Merged dataset path')
    parser_full.add_argument('--repo-id', required=True, help='Dataset repo_id')
    parser_full.add_argument('--left-dim', type=int, default=7, help='Left arm dimension (default: 7)')
    parser_full.add_argument('--right-dim', type=int, default=7, help='Right arm dimension (default: 7)')
    parser_full.add_argument('--num-workers', type=int, default=4, help='Number of parallel worker processes (default: 4)')
    parser_full.add_argument('--flip-views', nargs='+', default=None,
                             help='View names to flip directly')
    parser_full.add_argument('--swap-left-view', type=str, default='observation.images.hand_left',
                             help='Left view to swap with right')
    parser_full.add_argument('--swap-right-view', type=str, default='observation.images.hand_right',
                             help='Right view to swap with left')
    parser_full.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')
    parser_full.add_argument('--robot-type', type=str, default='agilex', help='Robot type (default: agilex)')
    parser_full.add_argument('--features-json', type=str, default=None, help='Path to features.json file')
    parser_full.add_argument('--force', action='store_true', help='Force merge (ignore conflicts)')
    
    args = parser.parse_args()
    
    if args.command == 'create-mirror':
        create_mirror_dataset(
            args.src_path,
            args.tgt_path,
            args.left_dim,
            args.right_dim,
            args.num_workers,
            flip_views=args.flip_views,
            swap_left_view=args.swap_left_view,
            swap_right_view=args.swap_right_view,
        )
    
    elif args.command == 'merge':
        if not MERGE_AVAILABLE:
            print(f"Error: Merge script not found: {_MERGE_SCRIPT}")
            sys.exit(1)
        
        features = None
        if args.features_json:
            with open(args.features_json, 'r', encoding='utf-8') as f:
                features = json.load(f)
        
        merge_lerobot_datasets(
            args.src_paths,
            args.tgt_path,
            args.repo_id,
            args.fps,
            args.robot_type,
            features,
            args.force
        )
    
    elif args.command == 'full':
        print("=" * 60)
        print("Space Mirror Full Pipeline")
        print("=" * 60)
        print()
        
        # Step 1: Create mirrored dataset
        print("Step 1/2: Creating mirrored dataset")
        create_mirror_dataset(
            args.src_path,
            args.mirror_path,
            args.left_dim,
            args.right_dim,
            args.num_workers,
            flip_views=args.flip_views,
            swap_left_view=args.swap_left_view,
            swap_right_view=args.swap_right_view,
        )
        print()
        
        # Step 2: Merge datasets
        print("Step 2/2: Merging datasets")
        if not MERGE_AVAILABLE:
            print(f"Error: Merge script not found: {_MERGE_SCRIPT}")
            sys.exit(1)
        
        features = None
        if args.features_json:
            with open(args.features_json, 'r', encoding='utf-8') as f:
                features = json.load(f)
        
        merge_lerobot_datasets(
            [args.src_path, args.mirror_path],
            args.merge_path,
            args.repo_id,
            args.fps,
            args.robot_type,
            features,
            args.force
        )
        print()
        print("=" * 60)
        print("✓ All processing complete!")
        print("=" * 60)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()