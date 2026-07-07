# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""Real Time Chunking (RTC) configuration classes.

Based on:
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
"""

from dataclasses import dataclass
from enum import Enum


class RTCAttentionSchedule(str, Enum):
    """Attention schedule for RTC prefix weight construction."""

    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"


@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an
    inpainting problem, strategically handling overlapping timesteps
    between action chunks using prefix attention.
    """

    enabled: bool = False  # Default OFF — zero impact on existing training/inference

    # Core RTC settings
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        if self.max_guidance_weight <= 0:
            raise ValueError(
                f"max_guidance_weight must be positive, got {self.max_guidance_weight}"
            )
        if self.debug_maxlen <= 0:
            raise ValueError(
                f"debug_maxlen must be positive, got {self.debug_maxlen}"
            )
