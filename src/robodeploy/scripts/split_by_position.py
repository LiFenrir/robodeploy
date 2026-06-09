"""
Split LeRobot dataset by first frame observation.state Joint1 position.
Delete failure data (is_failure_data == 1), then split episodes into
low (Joint1 < 0.85) and high (Joint1 >= 0.85) groups.
Parquet files, videos, and internal episode_index are renumbered sequentially.
"""

import os
import shutil
import json
import pandas as pd
import numpy as np

SRC = "robodeploy/s1_data/lerobot"
OUT_LOW = "robodeploy/s1_data_low/lerobot"
OUT_HIGH = "robodeploy/s1_data_high/lerobot"
THRESHOLD = 0.85


def get_first_frame_joint1(parquet_path):
    """Get Joint1 from first frame of an episode."""
    df = pd.read_parquet(parquet_path)
    first = df[df["frame_index"] == 0]
    if len(first) == 0:
        return None
    state = first["observation.state"].iloc[0]
    return float(state[1])


def clean_parquet(src_path, dst_path, new_episode_index):
    """Remove failure rows, renumber indices, update episode_index, save."""
    df = pd.read_parquet(src_path)
    df_clean = df[df["is_failure_data"] == 0].copy()
    if len(df_clean) == 0:
        return 0
    df_clean = df_clean.reset_index(drop=True)
    df_clean["frame_index"] = df_clean.index
    df_clean["index"] = df_clean.index
    df_clean["episode_index"] = new_episode_index
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df_clean.to_parquet(dst_path, index=False)
    return len(df_clean)


def new_ep_filename(ep_idx):
    """Return episode filename for a given index."""
    return f"episode_{ep_idx:06d}"


def main():
    datasets = sorted(os.listdir(SRC))
    print(f"Found {len(datasets)} datasets: {datasets}")

    for out_dir in [OUT_LOW, OUT_HIGH]:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

    all_stats = {"low": {}, "high": {}}

    for ds_name in datasets:
        ds_path = os.path.join(SRC, ds_name)
        data_dir = os.path.join(ds_path, "data/chunk-000")
        videos_dir = os.path.join(ds_path, "videos/chunk-000")
        meta_dir = os.path.join(ds_path, "meta")

        if not os.path.isdir(data_dir):
            continue

        parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
        print(f"\n--- {ds_name}: {len(parquet_files)} episodes ---")

        # --- Pass 1: classify each episode into low/high ---
        classified = {"low": [], "high": []}  # list of (original_filename, joint1_val, n_frames_after_clean)

        for pf in parquet_files:
            src_pq = os.path.join(data_dir, pf)
            j1 = get_first_frame_joint1(src_pq)
            if j1 is None:
                continue
            group = "high" if j1 >= THRESHOLD else "low"
            classified[group].append({"orig_name": pf, "joint1": j1, "src_pq": src_pq})

        # --- Pass 2: copy with new sequential indices ---
        for group, out_base in [("low", OUT_LOW), ("high", OUT_HIGH)]:
            items = classified[group]
            if not items:
                continue

            out_data_dir = os.path.join(out_base, ds_name, "data/chunk-000")
            out_videos_dir = os.path.join(out_base, ds_name, "videos/chunk-000")
            out_meta_dir = os.path.join(out_base, ds_name, "meta")
            os.makedirs(out_data_dir, exist_ok=True)
            os.makedirs(out_meta_dir, exist_ok=True)

            stats = []
            new_idx = 0  # only increment for episodes with data

            for item in items:
                src_pq = item["src_pq"]
                j1 = item["joint1"]
                orig_name = item["orig_name"]
                orig_ep_name = os.path.splitext(orig_name)[0]

                n_frames = clean_parquet(src_pq, None, new_idx)  # check first

                if n_frames > 0:
                    new_name = new_ep_filename(new_idx) + ".parquet"
                    dst_pq = os.path.join(out_data_dir, new_name)
                    # Re-save with correct path (we already have the cleaned df, but
                    # clean_parquet already saved to None, so just redo it properly)
                    clean_parquet(src_pq, dst_pq, new_idx)

                    stats.append({"episode_index": new_idx, "length": n_frames})

                    # Copy and rename videos
                    if os.path.isdir(videos_dir):
                        for camera in os.listdir(videos_dir):
                            cam_path = os.path.join(videos_dir, camera)
                            if not os.path.isdir(cam_path):
                                continue
                            src_video = os.path.join(cam_path, f"{orig_ep_name}.mp4")
                            if os.path.exists(src_video):
                                dst_cam = os.path.join(out_videos_dir, camera)
                                os.makedirs(dst_cam, exist_ok=True)
                                dst_video = os.path.join(dst_cam, f"{new_ep_filename(new_idx)}.mp4")
                                if not os.path.exists(dst_video):
                                    shutil.copy2(src_video, dst_video)

                    print(f"  [{group.upper():4s}] {orig_name} → {new_name}  Joint1={j1:.4f}  frames={n_frames}")
                    new_idx += 1
                else:
                    print(f"  [{group.upper():4s}] {orig_name} → (skipped, all failure)  Joint1={j1:.4f}")

            # --- Meta files ---
            if not stats:
                continue

            # info.json
            src_info = os.path.join(meta_dir, "info.json")
            if os.path.exists(src_info):
                with open(src_info) as fi:
                    info = json.load(fi)
                info["total_episodes"] = len(stats)
                info["total_frames"] = sum(s["length"] for s in stats)
                info["splits"] = {"train": f"0:{len(stats)}"}
                with open(os.path.join(out_meta_dir, "info.json"), "w") as fo:
                    json.dump(info, fo, indent=2, ensure_ascii=False)

            # tasks.jsonl (task-level, copy as-is)
            src_tasks = os.path.join(meta_dir, "tasks.jsonl")
            if os.path.exists(src_tasks):
                shutil.copy2(src_tasks, os.path.join(out_meta_dir, "tasks.jsonl"))

            # episodes.jsonl
            ep_jsonl = os.path.join(out_meta_dir, "episodes.jsonl")
            with open(ep_jsonl, "w") as f:
                for s in stats:
                    rec = {"episode_index": s["episode_index"], "tasks": ["hang cloths"], "length": s["length"]}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # episodes_stats.jsonl
            lengths = [s["length"] for s in stats]
            ep_stats_path = os.path.join(out_meta_dir, "episodes_stats.jsonl")
            stats_data = {
                "count": len(lengths),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "sum": int(np.sum(lengths)),
                "percentiles": {
                    "10": float(np.percentile(lengths, 10)),
                    "25": float(np.percentile(lengths, 25)),
                    "50": float(np.percentile(lengths, 50)),
                    "75": float(np.percentile(lengths, 75)),
                    "90": float(np.percentile(lengths, 90)),
                },
            }
            with open(ep_stats_path, "w") as f:
                f.write(json.dumps(stats_data, ensure_ascii=False) + "\n")

            all_stats[group][ds_name] = {"episodes": len(stats), "frames": sum(s["length"] for s in stats)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for tag in ["low", "high"]:
        total_eps = sum(v["episodes"] for v in all_stats[tag].values())
        total_frames = sum(v["frames"] for v in all_stats[tag].values())
        cmp = ">=" if tag == "high" else "<"
        print(f"\n{tag.upper()} position (Joint1 {cmp} {THRESHOLD}):")
        for ds, v in all_stats[tag].items():
            if v["episodes"] > 0:
                print(f"  {ds}: {v['episodes']} episodes, {v['frames']} frames")
        print(f"  TOTAL: {total_eps} episodes, {total_frames} frames")


if __name__ == "__main__":
    main()
