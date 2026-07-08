# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""Real-Time Chunking (RTC) implementation.

Based on Physical Intelligence's Kinetix implementation:
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L214
"""

import logging
import math

import torch
from torch import Tensor

from .configuration_rtc import RTCAttentionSchedule, RTCConfig
from .debug_tracker import Tracker

logger = logging.getLogger(__name__)


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    Wraps a denoiser callable and applies prefix-attention guidance
    that pulls the new chunk toward the previous chunk's unexecuted tail.
    """

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config
        self.tracker = None
        if rtc_config.debug:
            self.tracker = Tracker(
                enabled=rtc_config.debug,
                maxlen=rtc_config.debug_maxlen,
            )

    # ------------------------------------------------------------------
    # Tracker proxy methods
    # ------------------------------------------------------------------

    def track(
        self,
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        if self.tracker is not None:
            self.tracker.track(
                time=time,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                **metadata,
            )

    def get_all_debug_steps(self) -> list:
        if self.tracker is not None:
            return self.tracker.get_all_steps()
        return []

    def is_debug_enabled(self) -> bool:
        return self.tracker is not None and self.tracker.enabled

    def reset_tracker(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

    # ------------------------------------------------------------------
    # Core RTC denoising step
    # ------------------------------------------------------------------

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ) -> Tensor:
        """RTC guidance wrapper around an existing denoiser.

        Args:
            x_t: Current latent  (B,T,A) or (T,A).
            prev_chunk_left_over: Unexecuted prefix from previous chunk, or None.
            inference_delay: Steps of inference delay (affects prefix weights).
            time: Scalar in [0,1] (flow matching convention: 1→0).
            original_denoise_step_partial: callable (x_t) → v_t.
            execution_horizon: Override for config.execution_horizon.

        Returns:
            Guided velocity with the same shape as v_t.
        """
        tau = 1 - time  # invert: RTC paper uses 0→1

        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)

        squeezed = False
        if x_t.ndim < 3:
            x_t = x_t.unsqueeze(0)
            squeezed = True
        if prev_chunk_left_over.ndim < 3:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        B, T, A = x_t.shape
        leftover_len = prev_chunk_left_over.shape[1]

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon
        # 约束窗口: [inference_delay, inference_delay + constraint_len)
        constraint_len = min(execution_horizon, leftover_len)
        if constraint_len <= 0:
            if squeezed:
                return original_denoise_step_partial(x_t).squeeze(0)
            return original_denoise_step_partial(x_t)
        constraint_end = inference_delay + constraint_len

        # Align prev_chunk_left_over shape to x_t: pad if smaller, truncate if larger
        pT, pA = prev_chunk_left_over.shape[1], prev_chunk_left_over.shape[2]
        if pT != T or pA != A:
            aligned = torch.zeros(B, T, A, device=x_t.device)
            copy_T, copy_A = min(pT, T), min(pA, A)
            aligned[:, :copy_T, :copy_A] = prev_chunk_left_over[:, :copy_T, :copy_A]
            prev_chunk_left_over = aligned

        weights = (
            self.get_prefix_weights(inference_delay, constraint_end, T)
            .to(x_t.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        v_t = original_denoise_step_partial(x_t)
        x1_t = x_t - time * v_t  # denoised prediction (v_t detached, so ∂x1_t/∂x_t = I)
        err = (prev_chunk_left_over - x1_t) * weights
        correction = err  # gradient is identity since v_t has no grad graph

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)
        tau_tensor = torch.as_tensor(tau)
        sq = (1 - tau_tensor) ** 2
        inv_r2 = (sq + tau_tensor**2) / sq
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction

        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)

        self.track(
            time=time,
            x1_t=x1_t,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=constraint_len,
        )

        return result

    # ------------------------------------------------------------------
    # Prefix weight schedules
    # ------------------------------------------------------------------

    def get_prefix_weights(self, start, end, total):
        if start > end:
            logger.warning("get_prefix_weights: start=%d > end=%d, clamping start to end", start, end)
            start = end
        schedule = self.rtc_config.prefix_attention_schedule

        if schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif schedule == RTCAttentionSchedule.LINEAR:
            lin = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif schedule == RTCAttentionSchedule.EXP:
            lin = self._linweights(start, end, total)
            lin = lin * torch.expm1(lin).div(math.e - 1)
            weights = self._add_trailing_zeros(lin, total, end)
            weights = self._add_leading_ones(weights, start, total)
        else:
            weights = torch.ones(total)

        return weights

    def _linweights(self, start, end, total):
        skip = max(total - end, 0)
        n = total - skip - start
        if end <= start or n <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, n + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total, end):
        n = total - end
        if n <= 0:
            return weights
        return torch.cat([weights, torch.zeros(n)])

    def _add_leading_ones(self, weights, start, total):
        n = min(start, total)
        if n <= 0:
            return weights
        return torch.cat([torch.ones(n), weights])
