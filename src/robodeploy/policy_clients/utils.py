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
"""Factory function for creating policy clients from configuration."""

from robodeploy.policy_clients.base import PolicyClient
from robodeploy.policy_clients.config import PolicyClientConfig


def make_policy_client_from_config(config: PolicyClientConfig) -> PolicyClient:
    """Create a policy client from a configuration object.

    Args:
        config: A PolicyClientConfig subclass instance.

    Returns:
        A PolicyClient implementation.

    Raises:
        ValueError: If the config type is not recognized.
    """
    if config.type == "openpi":
        from robodeploy.policy_clients.openpi import OpenPIPolicyClient

        return OpenPIPolicyClient(config)
    elif config.type == "lingbot":
        from robodeploy.policy_clients.lingbot import LingbotPolicyClient

        return LingbotPolicyClient(config)
    else:
        raise ValueError(f"Unknown policy client type: {config.type}")
