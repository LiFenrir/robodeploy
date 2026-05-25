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
"""Base configuration class for policy clients.

Uses draccus ChoiceRegistry for automatic config subclass dispatch.
"""

import abc
from dataclasses import dataclass

import draccus


@dataclass(kw_only=True)
class PolicyClientConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base configuration for policy inference clients.

    Subclasses must be registered with @PolicyClientConfig.register_subclass("type_name").
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
