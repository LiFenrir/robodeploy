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
"""Abstract base class for remote policy inference clients.

This module defines the PolicyClient interface for connecting to remote
policy inference services (e.g., OpenPI, WAM).

Example:
    ```python
    from lerobot_mini.policy_clients import PolicyClient
    from lerobot_mini.policy_clients.openpi import OpenPIClient

    client = OpenPIClient(host="localhost", port=8000)
    actions = client.infer(observation)
    ```
"""

import abc
from typing import Any


class PolicyClient(abc.ABC):
    """Abstract base class for remote policy inference clients.

    Implementations connect to remote policy servers via various protocols
    (WebSocket, HTTP, gRPC) and provide a unified interface for inference.

    Args:
        host: Server hostname or IP address.
        port: Server port number.
    """

    def __init__(self, host: str = "localhost", port: int | None = None):
        self.host = host
        self.port = port

    @abc.abstractmethod
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Run inference with an observation dict.

        Args:
            obs: Observation dictionary containing state, images, prompt, etc.
                 Format depends on the specific policy implementation.

        Returns:
            dict: Inference results, typically containing "actions" key.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the client state.

        Should be called when the environment is reset to clear any
        internal caches or stateful buffers.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_server_metadata(self) -> dict:
        """Get metadata from the connected server.

        Returns:
            dict: Server metadata such as model info, action space, etc.
        """
        raise NotImplementedError
