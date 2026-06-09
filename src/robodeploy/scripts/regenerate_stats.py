#!/usr/bin/env python3
"""Regenerate episodes_stats.jsonl for a LeRobot v2.1 dataset."""

import argparse
import json
import logging
import sys
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.robodeploy.datasets.compute_stats import compute_episode_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    ds = Path(args.dataset).resolve()
    with open(ds / "meta" / "info.json") as f:
        info = json.load(f)

    with jsonlines.open(ds / "meta" / "episodes.jsonl") as reader:
        episodes = list(reader)

    features = info.get("features", {})
    # Only compute stats for non-image/video features
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
                # Handle list/array columns (e.g. action, observation.state)
                if col.dtype == object and len(col) > 0 and isinstance(col.iloc[0], (list, np.ndarray)):
                    episode_data[key] = np.stack(col.to_numpy())
                else:
                    episode_data[key] = col.to_numpy()

        ep_stats = compute_episode_stats(episode_data, scalar_features)
        # Convert numpy arrays to lists for JSON serialization
        stats_json = {}
        for fkey, fstats in ep_stats.items():
            stats_json[fkey] = {}
            for skey, sval in fstats.items():
                if isinstance(sval, np.ndarray):
                    stats_json[fkey][skey] = sval.tolist()
                else:
                    stats_json[fkey][skey] = sval

        stats_entries.append({
            "episode_index": ep_idx,
            "stats": stats_json,
        })
        logger.info(f"  episode_{ep_idx:06d} stats computed")

    with jsonlines.open(ds / "meta" / "episodes_stats.jsonl", "w") as writer:
        writer.write_all(stats_entries)

    logger.info(f"Done. {len(stats_entries)} episode stats written to {ds / 'meta' / 'episodes_stats.jsonl'}")


if __name__ == "__main__":
    main()
