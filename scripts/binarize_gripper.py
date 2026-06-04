#!/usr/bin/env python3
"""
Binarize gripper joints (indices 6 and 13) in action and observation.state.
Threshold: < 0.2 → 0 (closed), >= 0.2 → 1 (open).
Recomputes episodes_stats.jsonl afterwards.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.robodeploy.datasets.compute_stats import compute_episode_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GRIPPER_INDICES = [6, 13]
THRESHOLD = 0.2


def binarize_column(table: pa.Table, col_name: str) -> pa.Table:
    """Binarize gripper values in a list column."""
    col = table.column(col_name)
    arr = col.to_pylist()
    new_arr = []
    for row in arr:
        vals = list(row)
        for gi in GRIPPER_INDICES:
            vals[gi] = 1.0 if vals[gi] >= THRESHOLD else 0.0
        new_arr.append(vals)
    new_col = pa.array(new_arr, type=col.type)
    idx = table.column_names.index(col_name)
    return table.set_column(idx, col_name, new_col)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    ds = Path(args.dataset).resolve()
    with open(ds / "meta" / "info.json") as f:
        info = json.load(f)

    with jsonlines.open(ds / "meta" / "episodes.jsonl") as reader:
        episodes = list(reader)

    target_cols = ["action", "observation.state"]

    # --- Binarize parquets ---
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // 1000
        pq_path = ds / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"

        table = pq.read_table(str(pq_path))
        for col_name in target_cols:
            if col_name in table.column_names:
                table = binarize_column(table, col_name)
        pq.write_table(table, str(pq_path))
        logger.info(f"  episode_{ep_idx:06d} binarized")

    # --- Recompute stats ---
    logger.info("Recomputing episode stats...")
    features = info.get("features", {})
    scalar_features = {
        k: v for k, v in features.items()
        if v.get("dtype") not in ("image", "video", "string")
    }

    stats_entries = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // 1000
        pq_path = ds / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
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

    with jsonlines.open(ds / "meta" / "episodes_stats.jsonl", "w") as writer:
        writer.write_all(stats_entries)

    logger.info(f"Done. {len(episodes)} episodes binarized, stats recomputed.")


if __name__ == "__main__":
    main()
