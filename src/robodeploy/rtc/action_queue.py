# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""Action queue management for Real-Time Chunking (RTC)."""

import logging
from threading import Lock

import torch
from torch import Tensor

from .configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    Args:
        cfg: RTC configuration controlling queue behavior.
    """

    def __init__(self, cfg: RTCConfig):
        self.queue: Tensor | None = None
        self.original_queue: Tensor | None = None
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None
            action = self.queue[self.last_index].clone()
            self.last_index += 1
            return action

    def clear(self) -> None:
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        with self.lock:
            return self.last_index

    def get_left_over(self) -> Tensor | None:
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].clone()

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ):
        with self.lock:
            delay = self._check_and_resolve_delays(
                real_delay, action_index_before_inference
            )
            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, delay)
            else:
                self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(
        self, original_actions: Tensor, processed_actions: Tensor, real_delay: int
    ):
        clamped = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        self.original_queue = original_actions[clamped:].clone()
        self.queue = processed_actions[clamped:].clone()
        logger.debug(
            "RTC replace — delay=%d clamped=%d q_shape=%s",
            real_delay, clamped, self.queue.shape,
        )
        self.last_index = 0

    def _append_actions_queue(
        self, original_actions: Tensor, processed_actions: Tensor
    ):
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
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        effective = max(0, real_delay)
        if action_index_before_inference is not None:
            diff = max(0, self.last_index - action_index_before_inference)
            if diff != real_delay:
                logger.warning(
                    "Delay mismatch: index_diff=%d real_delay=%d", diff, real_delay
                )
                return effective
        return effective
