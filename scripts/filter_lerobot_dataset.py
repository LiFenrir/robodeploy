#!/usr/bin/env python
"""
Filter a LeRobot v2.1 dataset by is_failure_data and/or is_infer_data criteria.

For each episode, reads the parquet to classify:
  - is_failure: True if majority of frames have is_failure_data=1
  - infer_type: true (pure_infer), false (pure_teleop), mixed (dagger)

Only episodes matching ALL specified filters are kept. Output is a valid LeRobot v2.1 dataset.

Usage:
    # Filter success episodes only
    python filter_lerobot_dataset.py \
        --dataset /data/bi_s1_0528_merged \
        --is-failure false \
        --output_dir /data \
        --repo_id bi_s1_0528_success

    # Filter pure inference episodes
    python filter_lerobot_dataset.py \
        --dataset /data/bi_s1_0528_merged \
        --is-infer true \
        --output_dir /data \
        --repo_id bi_s1_0528_pure_infer

    # Filter success + dagger episodes
    python filter_lerobot_dataset.py \
        --dataset /data/bi_s1_0528_merged \
        --is-failure false \
        --is-infer mixed \
        --output_dir /data \
        --repo_id bi_s1_0528_success_dagger
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import jsonlines
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_lerobot_datasets import (  # noqa: E402
    load_info,
    load_episodes,
    load_tasks,
    load_episodes_stats,
    get_video_keys,
    remap_parquet,
    copy_videos,
    build_merged_stats_entry,
    DEFAULT_CHUNK_SIZE,
    META_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def classify_episode(parquet_path: Path) -> tuple[bool, str]:
    """Return (is_failure: bool, infer_type: 'true'|'false'|'mixed') for an episode."""
    table = pq.read_table(parquet_path)
    n = table.num_rows
    cols = set(table.column_names)

    # Default values for missing columns
    fail_arr = table.column("is_failure_data").to_numpy() if "is_failure_data" in cols else None
    infer_arr = table.column("is_infer_data").to_numpy() if "is_infer_data" in cols else None

    is_failure = bool(fail_arr.mean() > 0.5) if fail_arr is not None else False

    if infer_arr is not None:
        n_infer = int((infer_arr == 1).sum())
        n_teleop = n - n_infer
    else:
        n_infer, n_teleop = 0, n

    if n_infer > 0 and n_teleop > 0:
        infer_type = "mixed"
    elif n_infer > 0:
        infer_type = "true"
    else:
        infer_type = "false"

    return is_failure, infer_type


def episode_matches(
    parquet_path: Path, want_failure: bool | None, want_infer: str | None
) -> bool:
    is_failure, infer_type = classify_episode(parquet_path)
    if want_failure is not None and is_failure != want_failure:
        return False
    if want_infer is not None and infer_type != want_infer:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Filter a LeRobot v2.1 dataset by is_failure_data / is_infer_data."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the source LeRobot dataset directory.")
    parser.add_argument("--is-failure", type=str, choices=["true", "false"], default=None,
                        help="Keep only success (false) or failure (true) episodes.")
    parser.add_argument("--is-infer", type=str, choices=["true", "false", "mixed"], default=None,
                        help="Keep only pure_infer (true), pure_teleop (false), or dagger (mixed) episodes.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output root directory.")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Output dataset name (subdirectory under --output_dir).")
    args = parser.parse_args()

    want_failure: bool | None = None if args.is_failure is None else (args.is_failure == "true")
    want_infer: str | None = args.is_infer

    ds_path = Path(args.dataset).resolve()
    if not (ds_path / "meta" / "info.json").exists():
        logger.error(f"Not a valid LeRobot dataset (missing meta/info.json): {ds_path}")
        sys.exit(1)

    if want_failure is None and want_infer is None:
        logger.error("At least one filter (--is-failure, --is-infer) must be specified.")
        sys.exit(1)

    # ---- Load metadata ----
    info = load_info(ds_path)
    episodes = load_episodes(ds_path)
    tasks = load_tasks(ds_path)
    stats = load_episodes_stats(ds_path)

    if not episodes:
        logger.warning("No episodes found in source dataset.")
        return

    features = info.get("features", {})
    video_keys = get_video_keys(features)
    has_videos = len(video_keys) > 0
    video_path_template = info.get("video_path", "")
    fps = info.get("fps", 30)
    codebase_version = info.get("codebase_version", "v2.1")
    robot_type = info.get("robot_type", "")

    # Build parquet column list (same order as merge script)
    default_feature_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    parquet_columns = [k for k in features if k not in video_keys]
    ordered_columns = sorted(
        parquet_columns,
        key=lambda k: (not k.startswith("action"), not k.startswith("observation."), k),
    )

    logger.info(f"Source dataset: {ds_path.name} | {len(episodes)} episodes, "
                f"{info.get('total_frames', '?')} frames")
    logger.info(f"Filters: is_failure={want_failure}, is_infer={want_infer}")

    # ---- Filter episodes ----
    kept = []  # list of (old_ep_idx, old_chunk)
    skipped = []
    for ep in episodes:
        old_ep_idx = ep["episode_index"]
        old_chunk = old_ep_idx // DEFAULT_CHUNK_SIZE
        src_parquet = (
            ds_path / f"data/chunk-{old_chunk:03d}/episode_{old_ep_idx:06d}.parquet"
        )
        if not src_parquet.exists():
            logger.warning(f"Missing parquet: {src_parquet}")
            skipped.append(old_ep_idx)
            continue

        if episode_matches(src_parquet, want_failure, want_infer):
            kept.append((old_ep_idx, old_chunk))
        else:
            skipped.append(old_ep_idx)

    if not kept:
        logger.warning("No episodes matched the filter criteria.")
        return

    logger.info(f"Kept {len(kept)} episodes, skipped {len(skipped)}")

    # ---- Write output ----
    output_root = Path(args.output_dir).resolve() / args.repo_id
    if output_root.exists():
        logger.warning(f"Output directory already exists: {output_root}")

    meta_dir = output_root / META_DIR
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Tasks: keep only referenced tasks
    kept_task_names = set()
    for old_ep_idx, _ in kept:
        ep = next(e for e in episodes if e["episode_index"] == old_ep_idx)
        if ep.get("tasks"):
            kept_task_names.add(ep["tasks"][0])

    remapped_tasks: dict[str, int] = {}
    for t in sorted(kept_task_names):
        remapped_tasks[t] = len(remapped_tasks)

    with jsonlines.open(meta_dir / "tasks.jsonl", "w") as tw:
        for task_name, tidx in sorted(remapped_tasks.items(), key=lambda x: x[1]):
            tw.write({"task_index": tidx, "task": task_name})

    # Process kept episodes
    new_ep_idx = 0
    global_frame = 0
    new_episodes = []
    new_stats_entries = []

    for old_ep_idx, old_chunk in kept:
        new_chunk = new_ep_idx // DEFAULT_CHUNK_SIZE
        ep = next(e for e in episodes if e["episode_index"] == old_ep_idx)
        task_name = ep["tasks"][0] if ep.get("tasks") else ""
        new_task_idx = remapped_tasks.get(task_name, 0)

        src_parquet = (
            ds_path / f"data/chunk-{old_chunk:03d}/episode_{old_ep_idx:06d}.parquet"
        )
        dst_parquet = (
            output_root / f"data/chunk-{new_chunk:03d}/episode_{new_ep_idx:06d}.parquet"
        )

        n_frames = remap_parquet(
            src_parquet, dst_parquet, new_ep_idx, new_task_idx, global_frame,
            ordered_columns, {},
        )

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
    total_chunks = (
        (total_episodes - 1) // DEFAULT_CHUNK_SIZE + 1 if total_episodes > 0 else 0
    )
    out_info = {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": global_frame,
        "total_tasks": len(remapped_tasks),
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
        json.dump(out_info, f, indent=2)

    logger.info(
        f"Filtered dataset: {total_episodes} episodes, {global_frame} frames, "
        f"{len(remapped_tasks)} tasks, {total_chunks} chunks"
    )
    logger.info(f"Output: {output_root}")


if __name__ == "__main__":
    main()
