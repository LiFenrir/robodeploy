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
"""OpenPI policy client configuration."""

from dataclasses import dataclass

from robodeploy.policy_clients.config import PolicyClientConfig


@PolicyClientConfig.register_subclass("openpi")
@dataclass(kw_only=True)
class OpenPIPolicyClientConfig(PolicyClientConfig):
    """Configuration for OpenPI WebSocket policy client."""

    host: str = "localhost"
    port: int = 8000

    # RTC (Real-Time Chunking) settings
    use_rtc: bool = False
    rtc_execution_horizon: int = 10
