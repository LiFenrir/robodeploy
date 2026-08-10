#!/usr/bin/env python3
"""
将 LeRobot v2.1 数据集中所有 episode 的 task 统一替换为指定描述。

输出会重写每个 parquet 的 task_index 为 0，并重新生成 tasks.jsonl、
episodes.jsonl 和 info.json，同时复制视频文件。
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

from robodeploy.datasets.utils import get_video_keys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="源数据集路径")
    parser.add_argument("--tgt", required=True, help="输出目录（脚本会在其下创建 --repo-id 子目录）")
    parser.add_argument("--repo-id", default="replaced_task", help="输出数据集名称")
    parser.add_argument("--task", required=True, help="新的 task 描述字符串")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    if not (src / "meta" / "info.json").exists():
        logger.error(f"不是 LeRobot 数据集: {src}")
        sys.exit(1)

    with open(src / "meta" / "info.json") as f:
        info = json.load(f)

    with jsonlines.open(src / "meta" / "episodes.jsonl") as reader:
        episodes = list(reader)

    features = info.get("features", {})
    video_keys = get_video_keys(features)
    total_episodes = len(episodes)

    tgt = Path(args.tgt).resolve() / args.repo_id
    if tgt.exists():
        logger.warning(f"输出目录已存在，删除: {tgt}")
        shutil.rmtree(tgt)

    new_episodes = []
    global_frame = 0

    for ep in episodes:
        old_idx = ep["episode_index"]
        new_idx = old_idx
        chunk = new_idx // 1000

        src_pq = src / f"data/chunk-{old_idx // 1000:03d}/episode_{old_idx:06d}.parquet"
        dst_pq = tgt / f"data/chunk-{chunk:03d}/episode_{new_idx:06d}.parquet"
        dst_pq.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(str(src_pq))
        n_frames = table.num_rows
        col_dict = {c: table.column(c) for c in table.column_names}
        col_dict["task_index"] = np.full(n_frames, 0, dtype=np.int64)
        pq.write_table(pa.table(col_dict), str(dst_pq))

        video_template = info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        for vk in video_keys:
            src_vid = src / video_template.format(
                episode_chunk=old_idx // 1000, video_key=vk, episode_index=old_idx
            )
            if src_vid.exists():
                dst_vid = tgt / video_template.format(
                    episode_chunk=chunk, video_key=vk, episode_index=new_idx
                )
                dst_vid.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_vid), str(dst_vid))

        new_episodes.append({
            "episode_index": new_idx,
            "tasks": [args.task],
            "length": ep["length"],
        })
        global_frame += n_frames

    meta_dir = tgt / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(meta_dir / "tasks.jsonl", "w") as tw:
        tw.write({"task_index": 0, "task": args.task})

    with jsonlines.open(meta_dir / "episodes.jsonl", "w") as ew:
        for ep in new_episodes:
            ew.write(ep)

    src_stats = src / "meta" / "episodes_stats.jsonl"
    if src_stats.exists():
        shutil.copy2(str(src_stats), str(meta_dir / "episodes_stats.jsonl"))

    total_chunks = (total_episodes - 1) // 1000 + 1 if total_episodes > 0 else 0
    new_info = {
        **info,
        "total_episodes": total_episodes,
        "total_frames": global_frame,
        "total_tasks": 1,
        "total_videos": total_episodes * len(video_keys),
        "total_chunks": total_chunks,
        "splits": {"train": f"0:{total_episodes}"},
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    logger.info(f"输出: {tgt}")
    logger.info(f"完成。{total_episodes} 个 episode，{global_frame} 帧，task 已统一替换。")


if __name__ == "__main__":
    main()
