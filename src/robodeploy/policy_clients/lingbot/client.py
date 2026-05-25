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
"""LingBot VLA policy inference client via WebSocket."""

import logging
import threading
import time
from typing import Any

import numpy as np
import websockets.sync.client

from robodeploy.policy_clients.base import PolicyClient
from robodeploy.policy_clients.lingbot.config import LingbotPolicyClientConfig
from robodeploy.policy_clients.openpi.msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


class LingbotPolicyClient(PolicyClient):
    """WebSocket client for LingBot VLA inference server.

    Connects to a LingBot server (deploy/lingbot_vla_policy.py) and sends
    raw observations with LeRobot-format keys. Image resizing and feature
    normalization are handled server-side.
    """

    def __init__(self, config: LingbotPolicyClientConfig):
        super().__init__(config.host, config.port)
        self._lock = threading.Lock()
        self._config = config
        self._server_metadata: dict = {}
        self._connected = False

        self._packer = Packer()
        self._uri = f"ws://{config.host}:{config.port}"

        try:
            self._ws = websockets.sync.client.connect(
                self._uri, compression=None, max_size=None
            )
            self._server_metadata = unpackb(self._ws.recv())
            self._connected = True
            logger.info(f"Connected to LingBot at {self._uri}")
        except ConnectionRefusedError:
            logger.warning(
                f"LingBot server not available at {self._uri}, "
                "policy inference disabled"
            )
        except Exception as e:
            logger.warning(f"Failed to connect to LingBot at {self._uri}: {e}")

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

        # Build observation dict with LeRobot-format keys.
        # LingBot server expects images as [H,W,C] uint8 RGB, state as float32.
        obs: dict[str, Any] = {}
        for cam_name, img in images.items():
            if img is not None:
                rgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
                obs[f"observation.images.{cam_name}"] = rgb

        obs["observation.state"] = np.asarray(state, dtype=np.float32)
        obs["task"] = prompt

        with self._lock:
            data = self._packer.pack(obs)
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            return unpackb(response)

    def reset(self) -> None:
        if not self._connected:
            return
        with self._lock:
            data = self._packer.pack(
                {"reset": True, "robo_name": self._config.robo_name}
            )
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            return unpackb(response)

    def get_server_metadata(self) -> dict:
        return self._server_metadata
