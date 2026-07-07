# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""Visualization utilities for RTC debug information."""

import numpy as np
import torch


class RTCDebugVisualizer:
    """Visualizer for RTC debug information (corrections, errors, weights, etc.)."""

    @staticmethod
    def plot_waypoints(
        axes,
        tensor,
        start_from: int = 0,
        color: str = "blue",
        label: str = "",
        alpha: float = 0.7,
        linewidth: float = 2,
        marker: str | None = None,
        markersize: int = 4,
    ):

        if tensor is None:
            return
        tensor_np = (
            tensor.detach().cpu().numpy()
            if isinstance(tensor, torch.Tensor)
            else tensor
        )
        if tensor_np.ndim == 3:
            tensor_np = tensor_np[0]
        elif tensor_np.ndim == 1:
            tensor_np = tensor_np.reshape(-1, 1)

        time_steps, num_dims = tensor_np.shape
        x_indices = np.arange(start_from, start_from + time_steps)

        num_axes = len(axes) if hasattr(axes, "__len__") else 1
        for dim_idx in range(min(num_dims, num_axes)):
            ax = axes[dim_idx] if hasattr(axes, "__len__") else axes
            if marker:
                ax.plot(
                    x_indices, tensor_np[:, dim_idx],
                    color=color, label=label if dim_idx == 0 else "",
                    alpha=alpha, linewidth=linewidth,
                    marker=marker, markersize=markersize,
                )
            else:
                ax.plot(
                    x_indices, tensor_np[:, dim_idx],
                    color=color, label=label if dim_idx == 0 else "",
                    alpha=alpha, linewidth=linewidth,
                )
            if not ax.xaxis.get_label().get_text():
                ax.set_xlabel("Step", fontsize=10)
            if not ax.yaxis.get_label().get_text():
                ax.set_ylabel(f"Dim {dim_idx}", fontsize=10)
            ax.grid(True, alpha=0.3)
