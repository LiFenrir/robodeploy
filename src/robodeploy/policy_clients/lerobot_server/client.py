"""LeRobot TCP server policy inference client.

Talks to pi05_server.py over raw TCP + pickle. All format conversion
(BGR→RGB, camera rename, state padding) happens client-side so the
server needs no modification.
"""

import logging
import socket
import threading
from typing import Any

import numpy as np

from robodeploy.policy_clients.base import PolicyClient
from robodeploy.policy_clients.lerobot_server.config import LeRobotServerPolicyClientConfig
from robodeploy.policy_clients.lerobot_server.protocol import recv_msg, send_msg

logger = logging.getLogger(__name__)


class LeRobotServerPolicyClient(PolicyClient):
    """TCP client for LeRobot inference server (pi05_server.py).

    Sends observations in the server's native pickle format and receives
    action predictions. Handles BGR→RGB conversion, camera renaming, and
    state dimension adaptation client-side.
    """

    def __init__(self, config: LeRobotServerPolicyClientConfig):
        super().__init__(config.host, config.port)
        self._lock = threading.Lock()
        self._config = config
        self._connected = False
        self._sock: socket.socket | None = None

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(15.0)
            self._sock.connect((config.host, config.port))
            self._connected = True
            logger.info(f"Connected to LeRobot server at {config.host}:{config.port}")
        except ConnectionRefusedError:
            logger.warning(
                f"LeRobot server not available at {config.host}:{config.port}, "
                "policy inference disabled"
            )
        except Exception as e:
            logger.warning(f"Failed to connect to LeRobot server at {config.host}:{config.port}: {e}")

    @property
    def connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        """Cleanly close the TCP connection (e.g. when switching away from POLICY mode)."""
        self.close()
        logger.info("LeRobot policy client disconnected.")

    def _ensure_connected(self) -> None:
        """Reconnect if the connection was previously closed."""
        if self._connected:
            return
        logger.info("Reconnecting LeRobot policy client to %s:%d...",
                    self._config.host, self._config.port)
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(15.0)
            self._sock.connect((self._config.host, self._config.port))
            self._connected = True
            logger.info("Reconnected to LeRobot server.")
        except Exception as e:
            logger.warning("Failed to reconnect to LeRobot server: %s", e)
            self._sock = None
            self._connected = False

    def infer(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        prompt: str = "",
    ) -> dict[str, Any]:
        """Run inference via TCP + pickle.

        Args:
            images: Dict of camera_name → BGR image (H, W, C) uint8.
            state: Joint positions as float32 array.
            prompt: Task description string.

        Returns:
            dict with "actions" key containing the predicted action as 2D array [1, action_dim].
        """
        if not self._connected or self._sock is None:
            self._ensure_connected()
        if not self._connected or self._sock is None:
            return {}

        import cv2

        # 1. Convert images: BGR→RGB and rename cameras to server-expected names.
        payload_images: dict[str, np.ndarray] = {}
        for cam_name, img in images.items():
            if img is not None:
                rgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
                target_name = self._config.camera_rename.get(cam_name, cam_name)
                payload_images[target_name] = rgb

        # 2. Convert state → qpos: pad or truncate to server-expected state_dim.
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        qpos = np.zeros(self._config.state_dim, dtype=np.float32)
        n_copy = min(len(s), self._config.state_dim)
        qpos[:n_copy] = s[:n_copy]

        # 3. Build payload matching pi05_server.py's expected format.
        payload = {
            "qpos": qpos,
            "images": payload_images,
            "task": prompt,
        }

        with self._lock:
            try:
                send_msg(self._sock, payload)
                reply = recv_msg(self._sock)
            except (ConnectionError, BrokenPipeError, OSError) as e:
                logger.error(f"Connection to LeRobot server lost: {e}")
                self._connected = False
                self._sock = None
                return {}

        if not reply.get("ok", False):
            error_msg = reply.get("error", "unknown error")
            raise RuntimeError(f"LeRobot server inference error: {error_msg}")

        action = np.array(reply["action"], dtype=np.float32)
        # logger.info(
        #     "Infer: qpos=%s, images=%s, action=%s",
        #     qpos.round(3),
        #     {k: v.shape for k, v in payload_images.items()},
        #     action.round(3),
        # )
        # action from server is (T, action_dim) — a chunk over T timesteps.
        # Keep the full chunk for StreamActionBuffer temporal smoothing;
        # when smoothing is disabled, record_body_teaching takes action[0].
        return {"actions": action}

    def reset(self) -> None:
        """No-op: TCP connection is stateless."""
        pass

    def get_server_metadata(self) -> dict:
        """TCP protocol has no metadata exchange."""
        return {}

    def close(self) -> None:
        """Close the TCP connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._connected = False

    def __del__(self) -> None:
        self.close()
