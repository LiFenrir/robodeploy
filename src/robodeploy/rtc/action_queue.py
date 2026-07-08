# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""Fixed-size action queue with overlap blending for Real-Time Chunking (RTC).

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
    """Thread-safe fixed-size queue for action chunk management.

    Step-based refill trigger + fixed-size blend window.  Three core params:
      - queue_size: fixed capacity (= server action_chunk)
      - inference_step_interval: trigger new inference every N consumed steps
      - execution_horizon (from cfg): blend overlap + RTC guidance constraint window

    Args:
        cfg: RTC configuration (enabled, execution_horizon, ...).
        queue_size: Fixed queue capacity (= number of actions per inference).
        inference_step_interval: Trigger inference every N consumed steps.
    """

    def __init__(self, cfg: RTCConfig, queue_size: int = 25, inference_step_interval: int = 6):
        self.queue: Tensor | None = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg
        self.queue_size = queue_size
        self.inference_step_interval = inference_step_interval

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
        """Clear queue and reset consumption index."""
        with self.lock:
            self.queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """Remaining actions in queue."""
        with self.lock:
            if self.queue is None:
                return 0
            return max(0, len(self.queue) - self.last_index)

    def empty(self) -> bool:
        """True if queue exhausted."""
        return self.qsize() <= 0

    def needs_refill(self) -> bool:
        """True when step-based trigger fires or queue exhausted.

        Triggers inference every ``inference_step_interval`` consumed steps,
        covering the inference latency window before the queue runs dry.
        """
        with self.lock:
            if self.queue is None:
                return True
            if self.last_index >= len(self.queue):
                return True
            return self.last_index >= self.inference_step_interval

    def get_left_over(self) -> Tensor | None:
        """Unexecuted tail for RTC guidance (prev_chunk_left_over).

        Returns None when nothing to constrain — first inference or queue
        exhausted.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            return self.queue[self.last_index :].clone()

    # ------------------------------------------------------------------
    # Merge (chunk arrival)
    # ------------------------------------------------------------------

    def merge(self, actions: Tensor, execution_horizon: int):
        """Replace queue with new chunk, blending overlap region.

        Phases:
          1. Drop already-executed prefix (``self.last_index`` steps)
          2. Crossfade old queue tail with new chunk prefix
          3. Truncate to ``queue_size``

        Args:
            actions: New action chunk [T, A].
            execution_horizon: Blend overlap steps (= cfg.execution_horizon).
        """
        with self.lock:
            delay = self.last_index  # actual consumed steps since last merge

            # Drop already-executed prefix
            clamped = max(0, min(delay, len(actions)))
            new_queue = actions[clamped:].clone()

            # Overlap blending: crossfade old tail with new chunk prefix
            blend_n = execution_horizon
            if blend_n > 0 and self.queue is not None and self.last_index < len(self.queue):
                old_tail = self.queue[self.last_index :].clone()
                overlap = min(len(old_tail), len(new_queue), blend_n)
                if overlap > 0:
                    w_old = torch.linspace(1.0, 0.0, overlap, device=new_queue.device)
                    w_new = 1.0 - w_old
                    new_queue[:overlap] = (
                        w_old.unsqueeze(-1) * old_tail[:overlap]
                        + w_new.unsqueeze(-1) * new_queue[:overlap]
                    )
                    logger.debug(
                        "RTC blend — overlap=%d delay=%d clamped=%d q_shape=%s",
                        overlap, delay, clamped, new_queue.shape,
                    )

            # Fixed-size: truncate if needed (excess from faster-than-expected inference)
            if len(new_queue) > self.queue_size:
                new_queue = new_queue[:self.queue_size]

            self.queue = new_queue
            self.last_index = 0
