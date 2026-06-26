#!/usr/bin/env python3
"""Copy observation.state to action, then insert an initial zero-state frame
with grippers=1. The last original state frame is dropped to keep the same
frame count, so videos stay in sync.

state[t]  = zero_frame (joints=0, grippers=1)  for t=0
          = original_state[t-1]                 for t>=1
action[t] = original_state[t]                   for all t

Usage:
  python scripts/copy_state_to_action.py --src <src_dataset> --tgt <tgt_dataset>
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.robodeploy.datasets.compute_stats import compute_episode_stats


def _detect_gripper_indices(joint_keys: list[str]) -> list[int]:
    """Return indices of gripper joints from observation.state key names."""
    return [i for i, k in enumerate(joint_keys) if "gripper" in k]


def _detect_joint_dim(info: dict) -> int:
    """Return the state/action dimension from dataset metadata."""
    state_names = info.get("features", {}).get("observation.state", {}).get("names", [])
    if state_names:
        return len(state_names)
    # Fallback: infer from action shape
    action_shape = info.get("features", {}).get("action", {}).get("shape", [])
    if action_shape:
        return action_shape[0]
    raise ValueError("Cannot determine joint dimension from dataset metadata")


def make_zero_state(joint_dim: int, gripper_indices: list[int]) -> np.ndarray:
    """Create initial state: all joints 0, grippers 1 (open)."""
    state = np.zeros(joint_dim, dtype=np.float32)
    for gi in gripper_indices:
        state[gi] = 1.0
    return state


def process_episode(pq_path: Path, joint_dim: int, gripper_indices: list[int]) -> int:
    """Process one episode parquet in-place. Returns frame count (unchanged).

    state[t]  = zero_frame          for t=0
              = original_state[t-1] for t>=1
    action[t] = original_state[t]   for all t

    Frame count stays T — videos remain in sync.
    """
    table = pq.read_table(str(pq_path))
    df = table.to_pandas()

    original_states = df["observation.state"].to_numpy()
    n_frames = len(original_states)

    # state: [zero_frame, s0, s1, ..., s{T-2}] — last original state dropped
    zero_frame = make_zero_state(joint_dim, gripper_indices)
    new_states = np.empty(n_frames, dtype=object)
    new_states[0] = zero_frame
    for i in range(n_frames - 1):
        new_states[i + 1] = original_states[i].astype(np.float32)

    # action: [s0, s1, ..., s{T-1}] — same as original state
    new_actions = np.empty(n_frames, dtype=object)
    for i in range(n_frames):
        new_actions[i] = original_states[i].astype(np.float32)

    new_columns = {}
    for col_name in table.column_names:
        if col_name == "observation.state":
            new_columns[col_name] = pa.array(new_states, type=table.column(col_name).type)
        elif col_name == "action":
            new_columns[col_name] = pa.array(new_actions, type=table.column(col_name).type)
        else:
            # All other columns (timestamp, frame_index, etc.) stay unchanged
            new_columns[col_name] = table.column(col_name)

    new_table = pa.table(new_columns)
    pq.write_table(new_table, str(pq_path))
    return n_frames


def recompute_stats(ds_path: Path, episodes: list, info: dict, chunks_size: int = 1000):
    """Recompute episodes_stats.jsonl."""
    features = info.get("features", {})
    scalar_features = {
        k: v for k, v in features.items()
        if v.get("dtype") not in ("image", "video", "string")
    }

    stats_entries = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // chunks_size
        pq_path = ds_path / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        table = pq.read_table(str(pq_path))
        df = table.to_pandas()

        episode_data = {}
        for key in scalar_features:
            if key in df.columns:
                col = df[key]
                if col.dtype == object and len(col) > 0 and isinstance(col.iloc[0], (list, np.ndarray)):
                    episode_data[key] = np.stack(col.to_numpy())
                else:
                    episode_data[key] = col.to_numpy()

        ep_stats = compute_episode_stats(episode_data, scalar_features)
        stats_json = {}
        for fkey, fstats in ep_stats.items():
            stats_json[fkey] = {}
            for skey, sval in fstats.items():
                if isinstance(sval, np.ndarray):
                    stats_json[fkey][skey] = sval.tolist()
                else:
                    stats_json[fkey][skey] = sval
        stats_entries.append({"episode_index": ep_idx, "stats": stats_json})

    with jsonlines.open(ds_path / "meta" / "episodes_stats.jsonl", "w") as writer:
        writer.write_all(stats_entries)


def rebuild_index(ds_path: Path, episodes: list, chunks_size: int = 1000) -> int:
    """Rebuild global index after adding one frame per episode."""
    global_idx = 0
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // chunks_size
        pq_path = ds_path / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        table = pq.read_table(str(pq_path))
        df = table.to_pandas()

        n_frames = len(df)
        df["index"] = np.arange(global_idx, global_idx + n_frames, dtype=np.int64)
        df["frame_index"] = np.arange(0, n_frames, dtype=np.int64)
        pq.write_table(pa.Table.from_pandas(df), str(pq_path))

        ep["length"] = n_frames
        global_idx += n_frames

    return global_idx


def main():
    parser = argparse.ArgumentParser(
        description="Copy state to action, prepend zero-state frame, shift state down."
    )
    parser.add_argument("--src", required=True, help="Source dataset path")
    parser.add_argument("--tgt", required=True, help="Target dataset path")
    args = parser.parse_args()

    src_path = Path(args.src).resolve()
    tgt_path = Path(args.tgt).resolve()

    if tgt_path.exists():
        print(f"Target already exists, removing: {tgt_path}")
        shutil.rmtree(tgt_path)

    # Copy full dataset first (parquet + videos + meta)
    print(f"Copying {src_path} -> {tgt_path} ...")
    shutil.copytree(src_path, tgt_path, symlinks=True)

    # Load metadata from target
    with open(tgt_path / "meta" / "info.json") as f:
        info = json.load(f)

    chunks_size = info.get("chunks_size", 1000)
    joint_dim = _detect_joint_dim(info)
    state_names = info.get("features", {}).get("observation.state", {}).get("names", [])
    gripper_indices = _detect_gripper_indices(state_names)
    print(f"Detected: joint_dim={joint_dim}, gripper_indices={gripper_indices}, chunks_size={chunks_size}")

    with jsonlines.open(tgt_path / "meta" / "episodes.jsonl") as reader:
        episodes = list(reader)

    # Process each episode
    total_frames = 0
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // chunks_size
        pq_path = tgt_path / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        new_len = process_episode(pq_path, joint_dim, gripper_indices)
        total_frames += new_len
        print(f"  episode_{ep_idx:06d}: {new_len} frames")

    # Rebuild global index
    print("Rebuilding global index...")
    total = rebuild_index(tgt_path, episodes, chunks_size)

    # Update episodes.jsonl
    with jsonlines.open(tgt_path / "meta" / "episodes.jsonl", "w") as writer:
        writer.write_all(episodes)

    # Update info.json
    info["total_episodes"] = len(episodes)
    info["total_frames"] = total
    with open(tgt_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Recompute stats
    print("Recomputing stats...")
    recompute_stats(tgt_path, episodes, info, chunks_size)

    print(f"\nDone. {len(episodes)} episodes, {total} total frames.")
    print(f"Output: {tgt_path}")


if __name__ == "__main__":
    main()
