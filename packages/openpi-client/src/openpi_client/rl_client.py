"""RL training message helpers for the openpi_client WebSocket protocol.

These functions extend the existing ``WebsocketClientPolicy``
send/recv pattern with reward, done, and reset fields needed for
online RL training (Stage 2).

Protocol (msgpack + numpy over WebSocket):

Robot PC → Training PC:
    {
        "state": np.ndarray [action_dim],
        "images": {camera_name: np.ndarray [H, W, 3]},
        "prompt": str,
        "reward": float,
        "done": bool,
        "success": bool,
        "intervention": bool,          # True if human teleop was active
        "action": np.ndarray [d],       # single-frame human action (intervention only)
        "rl_active": bool,             # True if RL mode is active
    }

Training PC → Robot PC:
    {
        "actions": np.ndarray [C, action_dim],       # actor actions (RL mode)
        "vla_actions": np.ndarray [C, action_dim],   # VLA reference actions
        "reset": bool,
    }
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Robot → Training: merge robot observation with RL metadata
# ---------------------------------------------------------------------------


def pack_rl_observation(
    observation: dict[str, Any],
    reward: float = 0.0,
    done: bool = False,
    success: bool = False,
    intervention: bool = False,
    action: np.ndarray | None = None,
    rl_active: bool = True,
) -> dict[str, Any]:
    """Build the message dict for an RL step.

    Args:
        observation: Raw robot observation dict with at least ``"state"``,
            may include ``"images"`` and ``"prompt"``.
        reward: Cumulative reward from the previous action chunk.
        done: Whether the previous episode ended.
        success: Whether the episode ended successfully.
        intervention: Whether human teleop was active during the previous chunk.
        action: Actual executed action ``[d]`` (set when intervention=True).
        rl_active: Whether RL mode is active (True = actor controls, store transitions).

    Returns:
        Dict ready to be passed to ``WebsocketClientPolicy.infer()``.
    """
    msg: dict[str, Any] = {
        **observation,
        "reward": float(reward),
        "done": bool(done),
        "success": bool(success),
        "intervention": bool(intervention),
        "rl_active": bool(rl_active),
    }
    if action is not None:
        msg["action"] = np.asarray(action, dtype=np.float32)
    return msg


# ---------------------------------------------------------------------------
# Training → Robot: parse action chunk + control signals
# ---------------------------------------------------------------------------


def unpack_rl_response(
    response: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Parse the Training PC response.

    Args:
        response: Response dict from ``WebsocketClientPolicy.infer()``.

    Returns:
        ``(actions, vla_actions, reset)`` where:
        - ``actions``: actor actions ``[C, action_dim]`` for RL mode
        - ``vla_actions``: VLA reference actions ``[C, action_dim]`` for non-RL mode
        - ``reset``: ``True`` when the Training PC requests a robot reset.
    """
    actions = response.get("actions")
    vla_actions = response.get("vla_actions")
    reset = bool(response.get("reset", False))
    return actions, vla_actions, reset
