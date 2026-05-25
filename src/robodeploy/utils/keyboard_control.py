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
"""Keyboard control utilities for real-time data collection.

Non-blocking keypress reading and episode labeling for terminal-based
robot control interfaces.
"""

import logging
import sys

logger = logging.getLogger(__name__)


def get_keypress() -> str | None:
    """Non-blocking read of a single keypress from stdin.

    Returns:
        Single character string, or None if no key was pressed.
    """
    import select

    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None


def prompt_success_failure() -> int:
    """Ask user to label episode outcome via keyboard.

    Blocks until user presses a valid key.

    Returns:
        1 for success, 0 for failure, -1 for discard/cancel.
    """
    import select
    import termios
    import tty

    print()
    print("=" * 50)
    print("  Label: [1] Success  |  [0] Failure  |  [2] Discard")
    print("=" * 50)
    result = None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)
        while result is None:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                char = sys.stdin.read(1)
                if char == "1":
                    print("  => SUCCESS")
                    result = 1
                elif char == "0":
                    print("  => FAILURE")
                    result = 0
                elif char == "2":
                    print("  => DISCARDED")
                    result = -1
                else:
                    print(f"  Invalid: '{char}'")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return result
