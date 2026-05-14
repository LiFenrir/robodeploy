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
"""Configuration dataclass for record_s1_inference.py.

Uses draccus for CLI parsing with nested config support.
Example:
    python record_s1_inference.py \
        --robot.type=s1_follower \
        --robot.port=/dev/ttyUSB0 \
        --teleop.type=s1_leader \
        --teleop.port=/dev/ttyUSB1 \
        --policy.type=openpi \
        --policy.host=localhost \
        --policy.port=8000 \
        --task="fold the box"
"""

from dataclasses import dataclass, field
from typing import Literal

# Import config modules to trigger draccus ChoiceRegistry registration
from lerobot_mini.robots import (  # noqa: F401
    bi_s1_follower,
    bi_so100_follower,
    s1_follower,
    so100_follower,
)
from lerobot_mini.teleoperators import (  # noqa: F401
    bi_s1_leader,
    bi_so100_leader,
    s1_leader,
    so100_leader,
)
from lerobot_mini.policy_clients import (  # noqa: F401
    openpi,
)

from lerobot_mini.policy_clients import PolicyClientConfig
from lerobot_mini.robots import RobotConfig
from lerobot_mini.teleoperators import TeleoperatorConfig


@dataclass
class RecordConfig:
    """Configuration for unified data collection + inference script."""

    # Robot (draccus ChoiceRegistry, use --robot.type=...)
    robot: RobotConfig | None = None

    # Teleop (draccus ChoiceRegistry, use --teleop.type=...)
    teleop: TeleoperatorConfig | None = None

    # Policy client (draccus ChoiceRegistry, use --policy.type=...)
    policy: PolicyClientConfig | None = None

    # Output settings
    output_dir: str = "./s1_data"
    repo_id: str = "dataset"
    task: str = "fold the box"
    fps: int = 30
    episode_time_s: float = 120.0

    # Temporal smoothing
    use_temporal_smoothing: bool = True
    inference_rate: float = 3.0
    latency_k: int = 8
    min_smooth_steps: int = 8

    # Alignment
    align_max_step: float = 0.02

    # Control mode
    control_mode: Literal["teleop", "policy", "mixed"] = "mixed"
    control_mode_initial: Literal["teleop", "policy"] = "teleop"

    # WebUI
    webui_port: int = 8080
