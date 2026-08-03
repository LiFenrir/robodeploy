"""LeRobot TCP server policy client configuration."""

from dataclasses import dataclass, field

from robodeploy.policy_clients.config import PolicyClientConfig


@PolicyClientConfig.register_subclass("lerobot_server")
@dataclass(kw_only=True)
class LeRobotServerPolicyClientConfig(PolicyClientConfig):
    """Configuration for LeRobot TCP inference server client.

    Connects to a LeRobot pi05_server.py (or compatible) over raw TCP + pickle.
    """

    host: str = "localhost"
    port: int = 5005

    # Camera name mapping: robodeploy camera name → server-expected camera name.
    # The server (pi05_server.py) expects specific image keys like "top", "left_hand",
    # "right_hand". This mapping translates the robot's camera names.
    camera_rename: dict[str, str] = field(default_factory=lambda: {
        "top_head": "top",
        "hand_left": "left_hand",
        "hand_right": "right_hand",
    })

    # Dimensions expected by the server.
    state_dim: int = 14          # Pad/truncate state to this dim
    action_dim: int = 14         # Action dim received from server
