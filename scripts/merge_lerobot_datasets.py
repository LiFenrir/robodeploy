#!/usr/bin/env python
"""
Merge multiple LeRobot v2.1 datasets into one.

Walks each source dataset's episodes in order, remaps episode_index / index / task_index,
copies parquet files and videos, and regenerates all metadata.

Features are merged as the union of all source datasets. Missing parquet columns are filled
with defaults (specified via --defaults or --expert-data flag).

Usage:
    # Merge datasets with identical features
    python merge_lerobot_datasets.py \
        --datasets /data/ds1 /data/ds2 /data/ds3 \
        --output_dir /data/merged_output \
        --repo_id my_merged_dataset

    # Merge expert data (no is_failure_data/is_infer_data) with collected data
    python merge_lerobot_datasets.py \
        --datasets /data/expert_ds /data/collected_ds \
        --output_dir /data/merged_output \
        --expert-data
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
META_DIR = "meta"
INFO_FILE = "meta/info.json"
EPISODES_FILE = "meta/episodes.jsonl"
TASKS_FILE = "meta/tasks.jsonl"
EPISODES_STATS_FILE = "meta/episodes_stats.jsonl"

# Legacy → canonical field name mapping (record_s1_inference.py used old names before fix)
# Value transform: "invert" flips the value, "keep" passes through unchanged
_LEGACY_FIELD_MAP = {
    "is_success":    ("is_failure_data", "invert"),
    "is_inference":  ("is_infer_data",   "keep"),
}

# Pa dtype to normalized dtype string for info.json
_PA_DTYPE_TO_INFO = {
    "float": "float32",
    "double": "float64",
    "int32": "int32",
    "int64": "int64",
}


def load_info(ds_path: Path) -> dict:
    with open(ds_path / INFO_FILE) as f:
        return json.load(f)


def load_episodes(ds_path: Path) -> list[dict]:
    path = ds_path / EPISODES_FILE
    if not path.exists():
        return []
    with jsonlines.open(path) as reader:
        return sorted(reader, key=lambda e: e["episode_index"])


def load_tasks(ds_path: Path) -> list[dict]:
    path = ds_path / TASKS_FILE
    if not path.exists():
        return []
    with jsonlines.open(path) as reader:
        return sorted(reader, key=lambda t: t["task_index"])


def _normalize_stats_entry(item: dict) -> dict:
    """Convert flat stats (record_s1_inference.py format) to lerobot standard nested format.

    Flat:  {"episode_index": 0, "state_mean": [...], "action_mean": [...], ...}
    Nested: {"episode_index": 0, "stats": {"action": {"min": [...], ...}, "observation.state": {...}, ...}}
    """
    if "stats" in item:
        return item  # Already in standard format

    stats = {}
    length = item.get("length", 0)
    for prefix, feat_key in [("state", "observation.state"), ("action", "action")]:
        feat_stats = {}
        for stat_name in ("mean", "std", "min", "max"):
            flat_key = f"{prefix}_{stat_name}"
            if flat_key in item:
                feat_stats[stat_name] = item[flat_key]
        if length > 0:
            feat_stats["count"] = [length]
        if feat_stats:
            stats[feat_key] = feat_stats

    return {"episode_index": item["episode_index"], "stats": stats}


def load_episodes_stats(ds_path: Path) -> dict[int, dict]:
    path = ds_path / EPISODES_STATS_FILE
    if not path.exists():
        return {}
    result = {}
    with jsonlines.open(path) as reader:
        for item in reader:
            item = _normalize_stats_entry(item)
            result[item["episode_index"]] = item
    return result


def infer_feature_from_column(col_name: str, pa_type) -> dict:
    """Infer a feature entry from a parquet column. Used for columns missing from info.json."""
    pa_str = str(pa_type)
    dtype = _PA_DTYPE_TO_INFO.get(pa_str, "float32")
    if pa.types.is_list(pa_type) or pa.types.is_fixed_size_list(pa_type):
        if pa.types.is_fixed_size_list(pa_type):
            shape = [pa_type.list_size]
        else:
            shape = [-1]
        return {"dtype": dtype, "shape": shape, "names": None}
    return {"dtype": dtype, "shape": [1], "names": None}


def compute_merged_features(all_infos: list[dict], all_ds_paths: list[Path]) -> dict:
    """Compute the union of features with legacy→canonical field name normalization."""
    merged: dict = {}
    for ds_path, info in zip(all_ds_paths, all_infos):
        src_features = info.get("features", {})
        src_parquet_cols = _get_parquet_columns(ds_path)

        for key, feat in src_features.items():
            canonical_key, _ = _LEGACY_FIELD_MAP.get(key, (key, "keep"))
            if canonical_key not in merged:
                merged[canonical_key] = dict(feat)

        for col_name, pa_type in src_parquet_cols.items():
            canonical_key, _ = _LEGACY_FIELD_MAP.get(col_name, (col_name, "keep"))
            if canonical_key not in merged:
                inferred = infer_feature_from_column(col_name, pa_type)
                logger.info(f"Inferred feature '{canonical_key}' from parquet: {inferred}")
                merged[canonical_key] = inferred

    return merged


def _get_parquet_columns(ds_path: Path) -> dict[str, pa.DataType]:
    """Read the first parquet file to get column names and types."""
    data_dir = ds_path / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        return {}
    schema = pq.read_schema(parquet_files[0])
    return {name: schema.field(name).type for name in schema.names}


def get_video_keys(features: dict) -> list[str]:
    return [k for k, v in features.items() if v.get("dtype") == "video"]


def _column_default(name: str, n_frames: int, defaults: dict) -> pa.Array | None:
    """Return a default-filled array for a missing column, or None if no default known."""
    if name == "is_failure_data":
        val = defaults.get("is_failure_data", 0)
        return pa.array(np.full(n_frames, val, dtype=np.int64))
    if name == "is_infer_data":
        val = defaults.get("is_infer_data", 0)
        return pa.array(np.full(n_frames, val, dtype=np.int64))
    if name in defaults:
        val = defaults[name]
        return pa.array(np.full(n_frames, val))
    return None


def _resolve_source_column(table, dst_name: str, src_cols: set, defaults: dict) -> pa.Array | None:
    """Get a column array for dst_name from the source table, handling legacy names.

    Legacy mapping:
      is_success (source) → is_failure_data (target), value inverted (1 - val)
      is_inference (source) → is_infer_data (target), value kept
    """
    if dst_name in src_cols:
        return table.column(dst_name)

    # Try legacy names
    for legacy, (canonical, transform) in _LEGACY_FIELD_MAP.items():
        if canonical == dst_name and legacy in src_cols:
            arr = table.column(legacy).to_numpy()
            if transform == "invert":
                return pa.array(1 - arr)
            return pa.array(arr)

    return None


def remap_parquet(
    src: Path,
    dst: Path,
    new_episode_index: int,
    new_task_index: int,
    global_frame_start: int,
    target_columns: list[str],
    defaults: dict,
) -> int:
    """Rewrite a single episode parquet with remapped indices + filled missing columns."""
    table = pq.read_table(src)
    n_frames = table.num_rows
    src_cols = set(table.column_names)

    new_cols = {}
    for name in target_columns:
        if name == "episode_index":
            new_cols[name] = pa.array(np.full(n_frames, new_episode_index, dtype=np.int64))
        elif name == "index":
            new_cols[name] = pa.array(
                np.arange(global_frame_start, global_frame_start + n_frames, dtype=np.int64)
            )
        elif name == "task_index":
            new_cols[name] = pa.array(np.full(n_frames, new_task_index, dtype=np.int64))
        elif name == "timestamp":
            resolved = _resolve_source_column(table, name, src_cols, defaults)
            if resolved is not None:
                col = resolved
                if col.type == pa.float32():
                    new_cols[name] = col.cast(pa.float64())
                else:
                    new_cols[name] = col
            else:
                new_cols[name] = _column_default(name, n_frames, defaults)
        else:
            resolved = _resolve_source_column(table, name, src_cols, defaults)
            if resolved is not None:
                new_cols[name] = resolved
            else:
                default_arr = _column_default(name, n_frames, defaults)
                if default_arr is not None:
                    new_cols[name] = default_arr
                else:
                    logger.warning(f"Column '{name}' missing in {src}, no default available — skipping")

    new_table = pa.Table.from_pydict(new_cols)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst)
    return n_frames


def _resolve_video_src(
    ds_path: Path,
    video_key: str,
    old_chunk: int,
    old_ep_idx: int,
    video_path_template: str,
) -> Path | None:
    canonical = (
        ds_path
        / video_path_template.format(
            episode_chunk=old_chunk, video_key=video_key, episode_index=old_ep_idx
        )
    )
    if canonical.exists():
        return canonical

    cam_short = video_key.rsplit(".", 1)[-1]
    if cam_short != video_key:
        legacy = (
            ds_path
            / video_path_template.format(
                episode_chunk=old_chunk, video_key=cam_short, episode_index=old_ep_idx
            )
        )
        if legacy.exists():
            return legacy
    return None


def copy_videos(
    ds_path: Path,
    video_keys: list[str],
    old_ep_idx: int,
    new_ep_idx: int,
    new_chunk: int,
    output_root: Path,
    video_path_template: str,
) -> None:
    old_chunk = old_ep_idx // DEFAULT_CHUNK_SIZE
    for vk in video_keys:
        src = _resolve_video_src(ds_path, vk, old_chunk, old_ep_idx, video_path_template)
        if src is not None:
            dst = (
                output_root
                / video_path_template.format(
                    episode_chunk=new_chunk, video_key=vk, episode_index=new_ep_idx
                )
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def build_merged_stats_entry(ep_stats: dict | None, new_ep_idx: int) -> dict | None:
    if ep_stats is None:
        return None
    entry = dict(ep_stats)
    entry["episode_index"] = new_ep_idx
    return entry


def main():
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot v2.1 datasets.")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="Paths to source dataset directories.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the merged dataset.")
    parser.add_argument("--repo_id", type=str, default="merged_dataset",
                        help="Dataset name (subdirectory under output_dir).")
    parser.add_argument("--expert-data", action="store_true",
                        help="Shortcut for --defaults '{\"is_failure_data\": 0, \"is_infer_data\": 0}'.")
    parser.add_argument("--defaults", type=str, default="{}",
                        help="JSON dict of default values for missing columns, e.g. '{\"is_failure_data\":0}'.")
    args = parser.parse_args()

    ds_paths = [Path(p).resolve() for p in args.datasets]
    for p in ds_paths:
        if not (p / INFO_FILE).exists():
            logger.error(f"Not a valid LeRobot dataset (missing {INFO_FILE}): {p}")
            sys.exit(1)

    output_root = Path(args.output_dir).resolve() / args.repo_id
    if output_root.exists():
        logger.warning(f"Output directory already exists: {output_root}")

    # ---- Load metadata ----
    all_infos = [load_info(p) for p in ds_paths]
    all_episodes = [load_episodes(p) for p in ds_paths]
    all_tasks = [load_tasks(p) for p in ds_paths]
    all_stats = [load_episodes_stats(p) for p in ds_paths]

    # ---- Parse defaults ----
    column_defaults: dict = json.loads(args.defaults)
    if args.expert_data:
        column_defaults.setdefault("is_failure_data", 0)
        column_defaults.setdefault("is_infer_data", 0)

    # ---- Compute merged features (union) ----
    features = compute_merged_features(all_infos, ds_paths)
    video_keys = get_video_keys(features)
    has_videos = len(video_keys) > 0

    # Use the first dataset's metadata as baseline
    video_path_template = all_infos[0].get("video_path", "")
    fps = all_infos[0].get("fps", 30)
    codebase_version = all_infos[0].get("codebase_version", "v2.1")
    robot_type = all_infos[0].get("robot_type", "")

    # Build ordered column list (video keys excluded from parquet; DEFAULT_FEATURES always present)
    default_feature_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    parquet_columns = [k for k in features if k not in video_keys]
    # Ensure default keys come first in standard order
    ordered_columns = sorted(parquet_columns, key=lambda k: (
        not k.startswith("action"),
        not k.startswith("observation."),
        k,
    ))

    logger.info(f"Datasets: {len(ds_paths)}")
    logger.info(f"Total source episodes: {sum(info['total_episodes'] for info in all_infos)}")
    logger.info(f"Video keys: {video_keys}")
    logger.info(f"Parquet columns: {ordered_columns}")
    if column_defaults:
        logger.info(f"Column defaults: {column_defaults}")

    # ---- Merge tasks ----
    merged_task_index: dict[str, int] = {}
    for tasks in all_tasks:
        for t in tasks:
            task_name = t["task"]
            if task_name not in merged_task_index:
                merged_task_index[task_name] = len(merged_task_index)

    meta_dir = output_root / META_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(meta_dir / "tasks.jsonl", "w") as tw:
        for task_name, tidx in sorted(merged_task_index.items(), key=lambda x: x[1]):
            tw.write({"task_index": tidx, "task": task_name})

    logger.info(f"Merged tasks: {len(merged_task_index)}")

    # ---- Process episodes ----
    new_ep_idx = 0
    global_frame = 0
    new_episodes = []
    new_stats_entries = []

    for ds_idx, (ds_path, episodes, stats) in enumerate(zip(ds_paths, all_episodes, all_stats)):
        logger.info(f"Processing {ds_path.name}: {len(episodes)} episodes")
        for ep in episodes:
            old_ep_idx = ep["episode_index"]
            old_chunk = old_ep_idx // DEFAULT_CHUNK_SIZE
            new_chunk = new_ep_idx // DEFAULT_CHUNK_SIZE
            task_name = ep["tasks"][0] if ep.get("tasks") else ""
            new_task_idx = merged_task_index.get(task_name, 0)

            src_parquet = (
                ds_path / f"data/chunk-{old_chunk:03d}/episode_{old_ep_idx:06d}.parquet"
            )
            dst_parquet = (
                output_root / f"data/chunk-{new_chunk:03d}/episode_{new_ep_idx:06d}.parquet"
            )
            if src_parquet.exists():
                n_frames = remap_parquet(
                    src_parquet, dst_parquet, new_ep_idx, new_task_idx, global_frame,
                    ordered_columns, column_defaults,
                )
            else:
                logger.warning(f"Missing parquet: {src_parquet}")
                n_frames = ep["length"]

            if has_videos:
                copy_videos(
                    ds_path, video_keys, old_ep_idx, new_ep_idx, new_chunk,
                    output_root, video_path_template,
                )

            new_episodes.append({
                "episode_index": new_ep_idx,
                "tasks": [task_name],
                "length": n_frames,
            })

            old_stat = stats.get(old_ep_idx)
            merged_stat = build_merged_stats_entry(old_stat, new_ep_idx)
            if merged_stat is not None:
                new_stats_entries.append(merged_stat)

            global_frame += n_frames
            new_ep_idx += 1

        logger.info(f"  Done. Cumulative: {new_ep_idx} episodes, {global_frame} frames")

    # ---- Write episodes.jsonl ----
    with jsonlines.open(meta_dir / "episodes.jsonl", "w") as ew:
        for ep in new_episodes:
            ew.write(ep)

    # ---- Write episodes_stats.jsonl ----
    if new_stats_entries:
        with jsonlines.open(meta_dir / "episodes_stats.jsonl", "w") as sw:
            for entry in new_stats_entries:
                sw.write(entry)

    # ---- Write info.json ----
    total_episodes = new_ep_idx
    total_chunks = (total_episodes - 1) // DEFAULT_CHUNK_SIZE + 1 if total_episodes > 0 else 0
    info = {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": global_frame,
        "total_tasks": len(merged_task_index),
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"} if total_episodes > 0 else {},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": video_path_template if has_videos else None,
        "features": features,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Merged dataset: {total_episodes} episodes, {global_frame} frames, "
                f"{len(merged_task_index)} tasks, {total_chunks} chunks")
    logger.info(f"Output: {output_root}")


if __name__ == "__main__":
    main()
