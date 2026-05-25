# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""StreamActionBuffer for temporal smoothing of policy action chunks.

Used in real-time policy inference to smoothly blend overlapping action
chunks from consecutive inference calls.
"""

import threading
from collections import deque

import numpy as np


class StreamActionBuffer:
    """Sliding-window action chunk buffer with linear overlap blending.

    When a new action chunk arrives from the policy server, it overlaps
    with the tail of the previous chunk. The overlap region is blended
    using linear weights (100% old → 0% old) for smooth transitions.

    Args:
        state_dim: Dimension of the action vector (e.g., 14 for bimanual).
    """

    def __init__(self, state_dim: int = 14):
        self.lock = threading.Lock()
        self.state_dim = state_dim
        self.cur_chunk: deque = deque()
        self.k = 0
        self.last_action: np.ndarray | None = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int = 8, min_m: int = 8) -> None:
        """Integrate a new action chunk with temporal smoothing.

        Args:
            actions_chunk: New action chunk [N, state_dim].
            max_k: Maximum latency compensation steps to drop from front.
            min_m: Minimum overlap length for smoothing.
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]

            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return

            overlap_len = min(len(old_list), len(new_chunk))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_chunk, maxlen=None)
                self.k = 0
                return
            if len(old_list) > len(new_chunk):
                old_list = old_list[: len(new_chunk)]
                overlap_len = len(new_chunk)

            w_old = np.array([1.0]) if overlap_len == 1 else np.linspace(1.0, 0.0, overlap_len)
            w_new = 1.0 - w_old
            smoothed = [
                w_old[i] * np.asarray(old_list[i], dtype=float)
                + w_new[i] * np.asarray(new_chunk[i], dtype=float)
                for i in range(overlap_len)
            ]
            self.cur_chunk = deque([a.copy() for a in smoothed + new_chunk[overlap_len:]], maxlen=None)
            self.k = 0

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return the next action to execute.

        Returns:
            Action vector [state_dim,] or None if buffer is empty.
        """
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def clear(self) -> None:
        """Clear the buffer and reset state."""
        with self.lock:
            self.cur_chunk.clear()
            self.last_action = None
            self.k = 0
