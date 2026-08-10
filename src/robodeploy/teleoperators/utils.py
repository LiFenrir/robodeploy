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

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator


def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    if config.type == "bi_s1_leader":
        from .bi_s1_leader import BiS1Leader

        return BiS1Leader(config)
    elif config.type == "s1_leader":
        from .s1_leader import S1Leader

        return S1Leader(config)
    else:
        raise ValueError(config.type)
