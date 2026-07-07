#!/usr/bin/env python3
"""
Filter LeRobot dataset: keep only valid episodes, re-index, regenerate metadata.

Validation per episode:
  - parquet exists, readable, column count matches expected, rows == length
  - all video files exist and are readable by ffprobe
  - video frame count approximately matches episode length

Usage:
    python filter_valid_episodes.py --src <dataset> --tgt <output> --repo-id <name>
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000

from robodeploy.datasets.utils import get_video_keys  # noqa: E402


def _probe_video(path: str) -> dict | None:
    """Probe video with ffprobe, return {width, height, codec, nb_frames, fps} or None."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,nb_frames,r_frame_rate",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        s = data["streams"][0]
        num, den = s.get("r_frame_rate", "30/1").split("/")
        return {
            "width": s.get("width"),
            "height": s.get("height"),
            "codec": s.get("codec_name"),
            "nb_frames": int(s.get("nb_frames", 0)),
            "fps": float(num) / float(den),
        }
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        return None


def validate_episode(
    ds_path: Path,
    ep: dict,
    video_keys: list[str],
    expected_columns: list[str],
    video_path_template: str,
) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    ep_idx = ep["episode_index"]
    chunk = ep_idx // DEFAULT_CHUNK_SIZE
    length = ep.get("length", 0)

    # Check parquet
    pq_path = ds_path / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
    if not pq_path.exists():
        return False, f"parquet missing: {pq_path}"
    try:
        table = pq.read_table(pq_path)
    except Exception as e:
        return False, f"parquet read error: {e}"
    if table.num_rows != length:
        return False, f"parquet rows {table.num_rows} != length {length}"
    for col in expected_columns:
        if col not in table.column_names:
            return False, f"missing column: {col}"

    # Check videos
    for vk in video_keys:
        vpath = ds_path / video_path_template.format(
            episode_chunk=chunk, video_key=vk, episode_index=ep_idx
        )
        if not vpath.exists():
            return False, f"video missing: {vpath}"
        info = _probe_video(str(vpath))
        if info is None:
            return False, f"video unreadable: {vpath}"
        nb = info["nb_frames"]
        if nb > 0 and abs(nb - length) > 5:
            return False, f"video frames {nb} != length {length}: {vpath}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Filter LeRobot dataset, keep valid episodes.")
    parser.add_argument("--src", required=True, help="Source dataset path")
    parser.add_argument("--tgt", required=True, help="Output parent directory")
    parser.add_argument("--repo-id", default="filtered", help="Output dataset name")
    parser.add_argument("--max-frames-diff", type=int, default=5,
                        help="Max allowed diff between video frames and episode length")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    if not (src / "meta" / "info.json").exists():
        logger.error(f"Not a LeRobot dataset: {src}")
        sys.exit(1)

    # Load metadata
    with open(src / "meta" / "info.json") as f:
        info = json.load(f)

    episodes_path = src / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        logger.error("episodes.jsonl missing")
        sys.exit(1)
    with jsonlines.open(episodes_path) as reader:
        episodes = list(reader)

    stats_path = src / "meta" / "episodes_stats.jsonl"
    stats_map = {}
    if stats_path.exists():
        with jsonlines.open(stats_path) as reader:
            for item in reader:
                stats_map[item["episode_index"]] = item

    features = info.get("features", {})
    video_keys = get_video_keys(features)
    video_template = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")

    # Build expected parquet columns
    expected_columns = [k for k in features if k not in video_keys]
    # Also expect standard keys
    for k in ("episode_index", "index", "task_index", "timestamp", "frame_index"):
        if k not in expected_columns:
            expected_columns.append(k)

    logger.info(f"Source: {src.name}, {len(episodes)} episodes, {len(video_keys)} video views")
    logger.info(f"Parquet columns: {expected_columns}")
    logger.info(f"Video keys: {video_keys}")

    # Validate
    valid_eps = []
    invalid_eps = []
    for ep in episodes:
        ok, reason = validate_episode(src, ep, video_keys, expected_columns, video_template)
        if ok:
            valid_eps.append(ep)
        else:
            invalid_eps.append((ep["episode_index"], reason))
            logger.warning(f"  INVALID ep_{ep['episode_index']:06d}: {reason}")

    logger.info(f"Valid: {len(valid_eps)}, Invalid: {len(invalid_eps)}")
    if invalid_eps:
        for ep_idx, reason in invalid_eps:
            logger.info(f"  Removed ep_{ep_idx:06d}: {reason}")

    if not valid_eps:
        logger.error("No valid episodes!")
        sys.exit(1)

    # Setup output
    tgt = Path(args.tgt).resolve() / args.repo_id
    if tgt.exists():
        logger.warning(f"Output exists, removing: {tgt}")
        shutil.rmtree(tgt)
    tgt.mkdir(parents=True)

    # Copy and remap episodes
    new_ep_idx = 0
    global_frame = 0
    new_episodes = []
    new_stats = []

    for old_ep in valid_eps:
        old_idx = old_ep["episode_index"]
        old_chunk = old_idx // DEFAULT_CHUNK_SIZE
        new_chunk = new_ep_idx // DEFAULT_CHUNK_SIZE
        length = old_ep["length"]

        # Copy parquet with remapped indices
        src_pq = src / f"data/chunk-{old_chunk:03d}/episode_{old_idx:06d}.parquet"
        dst_pq = tgt / f"data/chunk-{new_chunk:03d}/episode_{new_ep_idx:06d}.parquet"
        dst_pq.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_pq)
        n_rows = table.num_rows
        col_dict = {}
        for col_name in table.column_names:
            col_dict[col_name] = table.column(col_name)
        col_dict["episode_index"] = np.full(n_rows, new_ep_idx, dtype=np.int64)
        col_dict["index"] = np.arange(global_frame, global_frame + n_rows, dtype=np.int64)
        # task_index: keep first task from old ep if present
        task_name = old_ep.get("tasks", [""])[0] if old_ep.get("tasks") else ""
        task_idx = 0  # Default; will be remapped after task extraction
        col_dict["task_index"] = np.full(n_rows, task_idx, dtype=np.int64)

        new_table = table.from_pydict(col_dict)
        pq.write_table(new_table, dst_pq)

        # Copy videos
        for vk in video_keys:
            src_vid = src / video_template.format(
                episode_chunk=old_chunk, video_key=vk, episode_index=old_idx
            )
            if src_vid.exists():
                dst_vid = tgt / video_template.format(
                    episode_chunk=new_chunk, video_key=vk, episode_index=new_ep_idx
                )
                dst_vid.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_vid), str(dst_vid))

        new_episodes.append({
            "episode_index": new_ep_idx,
            "tasks": [task_name],
            "length": length,
        })

        old_stat = stats_map.get(old_idx)
        if old_stat:
            ns = dict(old_stat)
            ns["episode_index"] = new_ep_idx
            new_stats.append(ns)

        global_frame += n_rows
        new_ep_idx += 1

    # Rebuild tasks (extract from valid episodes)
    task_names = sorted({ep.get("tasks", [""])[0] for ep in new_episodes if ep.get("tasks")})
    task_map = {name: i for i, name in enumerate(task_names)}
    # Update task_index in parquet
    for i, ep in enumerate(new_episodes):
        tname = ep["tasks"][0] if ep["tasks"] else ""
        tidx = task_map.get(tname, 0)
        pq_path = tgt / f"data/chunk-{i // DEFAULT_CHUNK_SIZE:03d}/episode_{i:06d}.parquet"
        if pq_path.exists():
            table = pq.read_table(pq_path)
            col_dict = {c: table.column(c) for c in table.column_names}
            col_dict["task_index"] = np.full(table.num_rows, tidx, dtype=np.int64)
            pq.write_table(table.from_pydict(col_dict), pq_path)

    # Write metadata
    meta_dir = tgt / "meta"
    meta_dir.mkdir(exist_ok=True)

    with jsonlines.open(meta_dir / "episodes.jsonl", "w") as w:
        for ep in new_episodes:
            w.write(ep)

    if new_stats:
        with jsonlines.open(meta_dir / "episodes_stats.jsonl", "w") as w:
            for s in new_stats:
                w.write(s)

    with jsonlines.open(meta_dir / "tasks.jsonl", "w") as w:
        for name, idx in sorted(task_map.items(), key=lambda x: x[1]):
            w.write({"task_index": idx, "task": name})

    total_chunks = (new_ep_idx - 1) // DEFAULT_CHUNK_SIZE + 1 if new_ep_idx > 0 else 0
    new_info = {
        **{k: v for k, v in info.items() if k not in (
            "total_episodes", "total_frames", "total_tasks", "total_videos",
            "total_chunks", "chunks_size", "splits",
        )},
        "total_episodes": new_ep_idx,
        "total_frames": global_frame,
        "total_tasks": len(task_map),
        "total_videos": new_ep_idx * len(video_keys),
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "splits": {"train": f"0:{new_ep_idx}"} if new_ep_idx > 0 else {},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    logger.info(f"Filtered dataset: {new_ep_idx}/{len(episodes)} episodes, {global_frame} frames")
    logger.info(f"Output: {tgt}")


if __name__ == "__main__":
    main()
