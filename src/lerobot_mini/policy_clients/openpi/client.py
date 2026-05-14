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
"""OpenPI policy inference client with image preprocessing."""

import logging
import threading
from typing import Any

import numpy as np

from lerobot_mini.policy_clients.base import PolicyClient
from lerobot_mini.policy_clients.openpi.config import OpenPIPolicyClientConfig

logger = logging.getLogger(__name__)


class OpenPIPolicyClient(PolicyClient):
    """Async inference client wrapping OpenPI WebSocket policy with image preprocessing."""

    def __init__(self, config: OpenPIPolicyClientConfig):
        from lerobot_mini.policy_clients.openpi import WebsocketClientPolicy

        self._lock = threading.Lock()
        self._policy = None
        self._connected = False
        self._config = config
        try:
            self._policy = WebsocketClientPolicy(config.host, config.port)
            self._connected = True
            logger.info(f"Connected to OpenPI at {config.host}:{config.port}")
        except ConnectionRefusedError:
            logger.warning(
                f"OpenPI server not available at {config.host}:{config.port}, "
                "policy inference disabled"
            )
        except Exception as e:
            logger.warning(f"Failed to connect to OpenPI at {config.host}:{config.port}: {e}")

    @property
    def connected(self) -> bool:
        return self._connected

    def infer(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        prompt: str = "",
    ) -> dict[str, Any]:
        if not self._connected:
            return {}

        import cv2
        from lerobot_mini.policy_clients.openpi import resize_with_pad

        rgb_images = {}
        for cam_name, img in images.items():
            if img is not None:
                rgb_images[cam_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        payload_images = {}
        for cam_name, img in rgb_images.items():
            payload_images[cam_name] = resize_with_pad(
                np.array([img]), 224, 224
            )[0].transpose(2, 0, 1)

        payload = {"state": state, "images": payload_images, "prompt": prompt}
        with self._lock:
            return self._policy.infer(payload)

    def reset(self) -> None:
        with self._lock:
            if self._policy is not None:
                self._policy.reset()

    def get_server_metadata(self) -> dict:
        if self._policy is not None:
            return self._policy.get_server_metadata()
        return {}
