#!/usr/bin/env python3
"""Delete specified episodes from a LeRobot dataset and re-index the remainder.

Removes episodes by index, renumbers all remaining episodes sequentially,
reorganizes parquet + video chunks, and regenerates metadata (info.json,
episodes.jsonl, episodes_stats.jsonl). A timestamped backup is created before
any modifications.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robodeploy.datasets.utils import get_video_keys


def delete_episodes(dataset_root: str | Path, episode_indices: list[int]):
    root = Path(dataset_root)
    to_remove = set(episode_indices)

    if not (root / "meta/info.json").exists():
        raise ValueError(f"数据集目录无效，未找到 meta/info.json: {root}")

    # 1. 备份到父级文件夹
    backup_name = f"{root.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    backup_dir = root.parent / backup_name
    shutil.copytree(root, backup_dir)
    print(f"已备份到: {backup_dir}")

    # 2. 加载元数据
    info = json.loads((root / "meta/info.json").read_text())
    data_path_tmpl = info["data_path"]
    video_path_tmpl = info.get("video_path")
    chunks_size = info["chunks_size"]
    video_keys = get_video_keys(info["features"])

    episodes = {}
    with jsonlines.open(root / "meta/episodes.jsonl") as reader:
        for item in reader:
            episodes[item["episode_index"]] = item

    episodes_stats = {}
    with jsonlines.open(root / "meta/episodes_stats.jsonl") as reader:
        for item in reader:
            episodes_stats[item["episode_index"]] = item

    all_indices = sorted(episodes.keys())
    invalid = to_remove - set(all_indices)
    if invalid:
        raise ValueError(f"无效的 episode 序号: {sorted(invalid)}")

    kept = [ep for ep in all_indices if ep not in to_remove]

    # 3. 创建临时目录，存放重新编序后的文件
    tmp_dir = root / ".tmp_reorder"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    new_episodes = []
    new_stats = []
    total_frames = 0
    total_videos_actual = 0

    for new_idx, old_idx in enumerate(kept):
        old_chunk = old_idx // chunks_size
        new_chunk = new_idx // chunks_size

        # 处理 parquet：读取、修改 episode_index 和 index、写入临时目录
        old_pq = root / data_path_tmpl.format(episode_chunk=old_chunk, episode_index=old_idx)
        new_pq = tmp_dir / data_path_tmpl.format(episode_chunk=new_chunk, episode_index=new_idx)
        new_pq.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(str(old_pq))
        df = table.to_pandas()
        ep_len = len(df)
        df["episode_index"] = new_idx
        df["index"] = np.arange(total_frames, total_frames + ep_len)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(new_pq))

        # 处理视频文件
        if video_path_tmpl:
            for vk in video_keys:
                old_vid = root / video_path_tmpl.format(
                    episode_chunk=old_chunk, video_key=vk, episode_index=old_idx
                )
                if old_vid.exists():
                    new_vid = tmp_dir / video_path_tmpl.format(
                        episode_chunk=new_chunk, video_key=vk, episode_index=new_idx
                    )
                    new_vid.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(old_vid, new_vid)
                    total_videos_actual += 1

        # 更新 episodes 和 episodes_stats 元数据
        ep = episodes[old_idx].copy()
        ep["episode_index"] = new_idx
        new_episodes.append(ep)

        st = episodes_stats[old_idx].copy()
        st["episode_index"] = new_idx
        new_stats.append(st)

        total_frames += ep_len

    # 4. 用临时目录中的 data 和 videos 替换原目录
    # Move old directories aside first so a failed copy doesn't leave the
    # dataset in a partially-deleted state.
    backups = {}
    for sub in ["data", "videos"]:
        src_sub = root / sub
        if src_sub.exists():
            bak_sub = root / f".{sub}_bak"
            shutil.move(str(src_sub), str(bak_sub))
            backups[sub] = bak_sub
    for sub in ["data", "videos"]:
        if (tmp_dir / sub).exists():
            try:
                shutil.copytree(tmp_dir / sub, root / sub)
            except Exception:
                # Restore the original directories on failure
                for s, bak in backups.items():
                    if bak.exists() and not (root / s).exists():
                        shutil.move(str(bak), str(root / s))
                raise
    # Clean up the backups
    for bak in backups.values():
        if bak.exists():
            shutil.rmtree(bak)

    # 5. 更新 info.json
    info["total_episodes"] = len(kept)
    info["total_frames"] = total_frames
    info["total_videos"] = total_videos_actual
    info["total_chunks"] = (len(kept) - 1) // chunks_size + 1 if kept else 0
    info["splits"] = {"train": f"0:{len(kept)}"} if kept else {}

    with open(root / "meta/info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    # 6. 更新 episodes.jsonl 和 episodes_stats.jsonl
    with jsonlines.open(root / "meta/episodes.jsonl", "w") as writer:
        writer.write_all(new_episodes)

    with jsonlines.open(root / "meta/episodes_stats.jsonl", "w") as writer:
        writer.write_all(new_stats)

    # 7. 清理临时目录
    shutil.rmtree(tmp_dir)

    print(
        f"删除完成，共删除 {len(to_remove)} 个 episode，剩余 {len(kept)} 个，总帧数 {total_frames}"
    )


def main():
    parser = argparse.ArgumentParser(description="删除 LeRobot 数据集中指定序号的数据")
    parser.add_argument("dataset_root", type=str, help="数据集根目录路径")
    parser.add_argument(
        "episode_indices", type=int, nargs="+", help="要删除的 episode 序号（可指定多个）"
    )
    args = parser.parse_args()
    delete_episodes(args.dataset_root, args.episode_indices)


if __name__ == "__main__":
    main()
