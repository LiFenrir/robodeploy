# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""Action queue with overlap blending for Real-Time Chunking (RTC).

Two-layer smoothing:
  1. RTC server-side guidance — new chunk denoised toward prev_chunk_left_over
  2. Overlap blending — old queue tail linearly crossfaded with new chunk prefix
"""

import logging
from threading import Lock

import torch
from torch import Tensor

from .configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe dual-queue for action chunk management.

    Maintains separate queues for original (model-space) and processed
    (execution-space) actions.  Supports both RTC and non-RTC modes.

    Args:
        cfg: RTC configuration.
    """

    def __init__(self, cfg: RTCConfig):
        self.queue: Tensor | None = None  # processed actions for execution
        self.original_queue: Tensor | None = None  # original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Consumption
    # ------------------------------------------------------------------

    def get(self) -> Tensor | None:
        """Pop next action [A,] or None if queue exhausted."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index].clone()
            self.last_index += 1
            return action

    # ------------------------------------------------------------------
    # Queue state queries
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear both queues and reset consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """Remaining actions in processed queue."""
        with self.lock:
            if self.queue is None:
                return 0
            return max(0, len(self.queue) - self.last_index)

    def empty(self) -> bool:
        """True if queue exhausted."""
        return self.qsize() <= 0

    def get_action_index(self) -> int:
        """Current consumption index."""
        with self.lock:
            return self.last_index

    def get_left_over(self) -> Tensor | None:
        """Unexecuted original tail for RTC guidance (prev_chunk_left_over).

        Returns None when nothing to constrain — first inference or queue exhausted.
        """
        with self.lock:
            if self.original_queue is None or self.last_index >= len(self.original_queue):
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        """Unexecuted processed tail (currently executing actions)."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            return self.queue[self.last_index :].clone()

    # ------------------------------------------------------------------
    # Merge (chunk arrival)
    # ------------------------------------------------------------------

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ):
        """Merge new actions into the queue.

        RTC enabled: replace queue, truncating delay steps + crossfade overlap.
        RTC disabled: append to queue tail.

        Args:
            original_actions: Raw model output [T, A].
            processed_actions: Post-processed actions for execution [T, A].
            real_delay: Inference delay steps to skip in the new chunk.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
    ):
        """Replace queue (RTC mode): truncate delay steps, then crossfade overlap."""
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        new_original = original_actions[clamped_delay:].clone()
        new_processed = processed_actions[clamped_delay:].clone()

        # Overlap blending: crossfade old tail with new chunk prefix
        eh = self.cfg.execution_horizon
        if eh > 0 and self.queue is not None and self.last_index < len(self.queue):
            old_tail = self.queue[self.last_index :].clone()
            overlap = min(len(old_tail), len(new_processed), eh)
            if overlap > 0:
                w_old = torch.linspace(1.0, 0.0, overlap, device=new_processed.device)
                w_new = 1.0 - w_old
                new_processed[:overlap] = (
                    w_old.unsqueeze(-1) * old_tail[:overlap] + w_new.unsqueeze(-1) * new_processed[:overlap]
                )

        self.original_queue = new_original
        self.queue = new_processed
        self.last_index = 0

    def _append_actions_queue(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
    ):
        """Append to queue (non-RTC mode): truncate consumed prefix, then cat."""
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_and_resolve_delays(
        self,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ) -> int:
        """Validate computed delay against actual consumed steps."""
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                logger.warning(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay

        return effective_delay
