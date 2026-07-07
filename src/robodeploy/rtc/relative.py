# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""Relative-action helpers for Real-Time Chunking (RTC).

Simplified standalone version for openpi — does not depend on
lerobot.processor. Only needed when the policy uses relative actions.
"""

from __future__ import annotations

import torch


def reanchor_relative_rtc_prefix(
    prev_actions_absolute: torch.Tensor,
    current_state: torch.Tensor,
    relative_mask: torch.Tensor | None = None,
    norm_scale: torch.Tensor | None = None,
    norm_offset: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert absolute leftover RTC prefix to model-space for relative-action policies.

    When using relative actions, the RTC prefix (previous chunk's unexecuted
    tail) is stored in absolute coordinates. Before feeding it back to the
    policy, this helper re-expresses those actions relative to the robot's
    current joint state and optionally normalizes them.

    Args:
        prev_actions_absolute: Absolute action prefix (T, A).
        current_state: Current robot joint state (A,) or (1, A).
        relative_mask: Boolean mask of same length as action_dim, True where
            the action dimension is relative to state.
        norm_scale: Optional normalization std (A,) for model-space output.
        norm_offset: Optional normalization mean (A,) for model-space output.
        device: Target device for the output tensor.

    Returns:
        Relative actions suitable as model input (T, A).
    """
    state = current_state.detach().cpu()
    if state.dim() == 1:
        state = state.unsqueeze(0)  # (1, A)

    action_cpu = prev_actions_absolute.detach().cpu()
    action_dim = action_cpu.shape[-1]

    # Build mask: by default, treat all pos-like dimensions as relative
    if relative_mask is None:
        relative_mask = torch.zeros(action_dim, dtype=torch.bool)
        # Heuristic: assume last half of dimensions could be gripper
        mid = action_dim // 2
        relative_mask[:mid] = True

    # Convert: prev_action_abs → prev_action_rel = prev_action_abs - state
    relative_actions = action_cpu.clone()
    if state.shape[-1] == action_dim:
        relative_actions[:, relative_mask] = (
            action_cpu[:, relative_mask] - state[:, relative_mask]
        )
    else:
        # state dim != action dim — only apply to overlapping dims
        n = min(state.shape[-1], action_dim)
        rel_mask_sub = relative_mask[:n]
        relative_actions[:, :n][:, rel_mask_sub] = (
            action_cpu[:, :n][:, rel_mask_sub] - state[:, :n][:, rel_mask_sub]
        )

    # Optional normalization: (x - mean) / std
    if norm_scale is not None:
        scale = norm_scale.detach().cpu()
        offset = norm_offset.detach().cpu() if norm_offset is not None else 0.0
        relative_actions = (relative_actions - offset) / (scale + 1e-8)

    return relative_actions.to(device)
