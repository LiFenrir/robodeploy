#!/usr/bin/env python3
"""Compare observation.state and action correlation across datasets.

For each dataset, loads the first episode and produces:
  1. Per-joint Pearson correlation between state[t] and action[t]
  2. Joint time-series plots (state vs action over time)

Usage:
  python scripts/compare_state_action.py <dataset1> <dataset2> ...
  python scripts/compare_state_action.py /home/kemove/VLA/datasets/bi_s1/bi_s1_0626_0142 /home/kemove/VLA/datasets/bi_s1/bi_s1_0624
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def load_first_episode(dataset_path: str) -> dict:
    """Load the first episode parquet file and metadata from a dataset."""
    root = Path(dataset_path)
    info = json.loads((root / "meta" / "info.json").read_text())

    # Find first parquet file
    data_dir = root / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_000000.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode_000000.parquet found in {data_dir}")
    table = pq.read_table(parquet_files[0])

    state = np.array(table.column("observation.state").to_pylist())  # (T, D)
    action = np.array(table.column("action").to_pylist())  # (T, D)
    joint_names = info["features"]["observation.state"].get("names", [f"dim_{i}" for i in range(state.shape[1])])

    return {
        "name": root.name,
        "state": state,
        "action": action,
        "joint_names": joint_names,
        "num_frames": state.shape[0],
        "robot_type": info.get("robot_type", "unknown"),
    }


def compute_correlations(data: dict) -> np.ndarray:
    """Per-joint Pearson correlation between state[t] and action[t].

    Returns NaN for joints where the variance is zero (e.g. grippers always at 1.0).
    """
    state = data["state"]
    action = data["action"]
    n_frames = min(len(state), len(action))
    corrs = np.full(state.shape[1], np.nan)
    for j in range(state.shape[1]):
        s_std = np.std(state[:n_frames, j])
        a_std = np.std(action[:n_frames, j])
        if s_std > 1e-10 and a_std > 1e-10:
            corrs[j] = np.corrcoef(state[:n_frames, j], action[:n_frames, j])[0, 1]
    return corrs


def _plot_single_dataset_timeseries(ds: dict, colors: list, output_path: str):
    """Plot state and action time-series for a single dataset, one subplot per joint."""
    joint_names = ds["joint_names"]
    short_names = [n.replace(".pos", "").replace("left_", "L_").replace("right_", "R_") for n in joint_names]
    num_joints = len(joint_names)

    cols = 4
    rows = int(np.ceil(num_joints / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = axes.flatten()

    n_frames = min(len(ds["state"]), len(ds["action"]))
    t = np.arange(n_frames)

    for j in range(num_joints):
        ax = axes[j]
        ax.plot(t, ds["state"][:n_frames, j], color=colors[0], linewidth=0.8, alpha=0.85, label="state" if j == 0 else None)
        ax.plot(t, ds["action"][:n_frames, j], color=colors[1], linewidth=0.8, alpha=0.65, linestyle="--", label="action" if j == 0 else None)
        ax.set_title(short_names[j], fontsize=10)
        ax.set_xlabel("frame" if j >= num_joints - cols else "")
        ax.tick_params(labelsize=7)

    for j in range(num_joints, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"State vs Action — {ds['name']} (first episode)", fontsize=13, y=1.06)
    fig.tight_layout(rect=[0.02, 0, 0.98, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Time-series plot saved to: {output_path}")


def plot_joint_timeseries(datasets: list[dict], output_dir: str):
    """Plot state and action time-series for each joint, one figure per dataset.
    Each figure shows state (实线) vs action (虚线) for that dataset only."""
    # Two distinct colors per figure: state=blue, action=orange
    for ds in datasets:
        tag = ds["name"][:30]
        out_path = str(Path(output_dir) / f"timeseries_{tag}.png")
        _plot_single_dataset_timeseries(ds, colors=["#1f77b4", "#ff7f0e"], output_path=out_path)


def plot_correlation_heatmap(datasets: list[dict], output_path: str):
    """Plot correlation heatmap: datasets x joints.

    Datasets with different joint dimensions use the smaller common set.
    """
    # Validate joint name consistency
    ref_names = datasets[0]["joint_names"]
    for ds in datasets[1:]:
        if ds["joint_names"] != ref_names:
            print(f"  WARNING: joint name mismatch between {datasets[0]['name']} and {ds['name']}")
            print(f"    Reference ({datasets[0]['name']}): {ref_names}")
            print(f"    Mismatch  ({ds['name']}): {ds['joint_names']}")
    num_joints = min(len(ds["joint_names"]) for ds in datasets)
    joint_names = datasets[0]["joint_names"]
    short_names = [n.replace(".pos", "").replace("left_", "L_").replace("right_", "R_") for n in joint_names]

    corr_matrix = np.full((len(datasets), num_joints), np.nan)
    for di, ds in enumerate(datasets):
        corrs = compute_correlations(ds)
        corr_matrix[di, :len(corrs)] = corrs[:num_joints]

    fig, ax = plt.subplots(figsize=(max(num_joints * 0.8, 12), max(len(datasets) * 1.1, 3.5)))
    im = ax.imshow(corr_matrix, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)

    # Annotate
    for di in range(len(datasets)):
        for j in range(num_joints):
            val = corr_matrix[di, j]
            label = f"{val:.3f}" if not np.isnan(val) else "N/A"
            ax.text(j, di, label, ha="center", va="center", fontsize=7)

    ax.set_xticks(range(num_joints))
    ax.set_xticklabels(short_names[:num_joints], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([ds["name"] for ds in datasets], fontsize=9)
    ax.set_title("State-Action Pearson Correlation (first episode)", fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout(rect=[0.05, 0.02, 0.95, 0.93])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Correlation heatmap saved to: {output_path}")


def print_correlation_table(datasets: list[dict]):
    """Print per-joint correlation table to stdout."""
    # Precompute correlations once per dataset
    cached_corrs = [compute_correlations(ds) for ds in datasets]

    joint_names = datasets[0]["joint_names"]
    short_names = [n.replace(".pos", "").replace("left_", "L_").replace("right_", "R_") for n in joint_names]

    # Header
    header = f"{'Joint':<18}" + "".join(f"{ds['name']:>12}" for ds in datasets)
    print(f"\n  {header}")
    print(f"  {'-' * (18 + 12 * len(datasets))}")

    for j, name in enumerate(short_names):
        vals = "".join(f"{corrs[j]:12.4f}" if j < len(corrs) and not np.isnan(corrs[j]) else f"{'N/A':>12}"
                       for corrs in cached_corrs)
        print(f"  {name:<18}{vals}")

    # Mean correlation (ignoring NaN)
    means = "".join(f"{np.nanmean(corrs):12.4f}" for corrs in cached_corrs)
    print(f"  {'-' * (18 + 12 * len(datasets))}")
    print(f"  {'MEAN':<18}{means}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare state-action correlation across LeRobot datasets (first episode only)"
    )
    parser.add_argument("datasets", nargs="+", help="Dataset paths")
    parser.add_argument("--output-dir", "-o", default="./compare_output", help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    datasets = []
    for path in args.datasets:
        print(f"Loading: {path}")
        ds = load_first_episode(path)
        print(f"  name={ds['name']}, robot={ds['robot_type']}, "
              f"frames={ds['num_frames']}, joints={len(ds['joint_names'])}")
        datasets.append(ds)

    # Validate dimensions
    if len(datasets) > 1:
        dims = {ds["name"]: ds["state"].shape[1] for ds in datasets}
        if len(set(dims.values())) > 1:
            print(f"\n  WARNING: Datasets have different state dimensions: {dims}")
            print("  Plots and tables will use the minimum common dimension.\n")

    # Print correlation table
    print_correlation_table(datasets)

    # Plots
    tag = "_vs_".join(ds["name"][:15] for ds in datasets)
    plot_correlation_heatmap(datasets, str(out_dir / f"corr_heatmap_{tag}.png"))
    plot_joint_timeseries(datasets, str(out_dir))

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
