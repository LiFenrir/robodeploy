#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from robodeploy.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_s1_follower")
@dataclass
class BiS1FollowerConfig(RobotConfig):
    """Configuration for bimanual S1 follower arms using the Theseus S1 SDK."""

    # Required fields first (no defaults)
    left_arm_port: str
    right_arm_port: str

    # Left arm configuration (with defaults)
    left_arm_version: str = "V2"
    left_end_effector: str = "gripper"

    # Right arm configuration (with defaults)
    right_arm_version: str = "V2"
    right_end_effector: str = "gripper"

    # Collision checking
    check_collision: bool = True

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Maximum gripper position (for gripper end effector)
    max_gripper_pos: float = 100.0
