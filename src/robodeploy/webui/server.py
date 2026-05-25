#!/usr/bin/env python
"""FastAPI-based WebUI server for lerobot-mini data collection.

Pure WebSocket architecture:
- JSON messages for status broadcast and commands
- Binary messages for video frames: [4B cam_name_len LE] + [cam_name UTF-8] + [JPEG data]

Runs in a background thread, sharing state with the main control loop
via state_ref/recording_ref/stop_ref dictionaries and an obs_lock.
"""

import asyncio
import io
import json
import logging
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable
from fastapi import WebSocket

import numpy as np

logger = logging.getLogger(__name__)


class WebUIServer:
    """WebUI server that runs in a background thread.

    Serves a touch-friendly frontend, camera frames over WebSocket binary,
    and bidirectional JSON control/status over the same WebSocket.
    """

    def __init__(
        self,
        state_ref: dict,
        recording_ref: dict,
        stop_ref: dict,
        obs_lock: threading.Lock,
        camera_names: list[str],
        port: int = 8080,
        command_handlers: dict[str, Callable] | None = None,
        fps: int = 30,
    ):
        self.state_ref = state_ref
        self.recording_ref = recording_ref
        self.stop_ref = stop_ref
        self.obs_lock = obs_lock
        self.camera_names = camera_names
        self.port = port
        self.command_handlers = command_handlers or {}
        self.fps = fps

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set = set()
        self._history: deque[dict] = deque(maxlen=20)
        self._start_time: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the server in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the server to shut down."""
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """Thread target: create event loop and start uvicorn."""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import RedirectResponse
        from fastapi.staticfiles import StaticFiles
        import uvicorn

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        static_dir = Path(__file__).with_suffix("").parent / "static"
        app = FastAPI()

        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        if static_dir.is_dir():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def root():
            return RedirectResponse(url="/static/index.html")

        @app.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            self._clients.add(websocket)

            async def broadcast_loop():
                try:
                    while True:
                        await self._broadcast_json(self._build_status())
                        await self._broadcast_video_frames()
                        interval = 1.0 / self.fps
                        await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            broadcast_task = asyncio.create_task(broadcast_loop())

            try:
                while True:
                    msg = await websocket.receive_text()
                    await self._handle_ws_message(websocket, msg)
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                broadcast_task.cancel()
                try:
                    await broadcast_task
                except asyncio.CancelledError:
                    pass
                self._clients.discard(websocket)

        config = uvicorn.Config(
            app, host="0.0.0.0", port=self.port, loop="none",
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._loop.run_until_complete(server.serve())

    async def _shutdown(self) -> None:
        """Close all WebSocket connections."""
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()

    # ------------------------------------------------------------------
    # Video frame broadcasting (WebSocket binary)
    # ------------------------------------------------------------------

    async def _broadcast_video_frames(self) -> None:
        """Send JPEG-encoded camera frames to all WebSocket clients as binary."""
        if not self._clients:
            return

        import cv2

        frames: dict[str, bytes] = {}
        with self.obs_lock:
            obs = self.state_ref.get("obs")
            if obs is None:
                return
            for cam_name in self.camera_names:
                frame = obs.get(cam_name)
                if frame is None:
                    continue
                try:
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    elif frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # imencode expects BGR
                    else:
                        # Assume RGB from robot cameras, convert to BGR for imencode
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        data = encoded.tobytes()
                        name_bytes = cam_name.encode("utf-8")
                        header = struct.pack("<I", len(name_bytes))
                        frames[cam_name] = header + name_bytes + data
                except Exception as exc:
                    logger.debug("Frame encode error for %s: %s", cam_name, exc)

        if not frames:
            return

        dead = set()
        for ws in self._clients:
            try:
                for cam_name in self.camera_names:
                    if cam_name in frames:
                        await ws.send_bytes(frames[cam_name])
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ------------------------------------------------------------------
    # JSON broadcasting
    # ------------------------------------------------------------------

    async def _broadcast_json(self, payload: dict) -> None:
        """Send a JSON payload to all connected WebSocket clients."""
        dead = set()
        for ws in self._clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ------------------------------------------------------------------
    # WebSocket command handling
    # ------------------------------------------------------------------

    async def _handle_ws_message(self, websocket: WebSocket, msg: str) -> None:
        """Parse and dispatch a WebSocket command."""
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            await websocket.send_json({"error": "invalid json"})
            return

        cmd = data.get("cmd")
        handler = self.command_handlers.get(cmd)
        if handler is None:
            await websocket.send_json({"error": f"unknown cmd: {cmd}"})
            return

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(data)
            else:
                result = handler(data)
            if result is not None:
                await websocket.send_json({"ack": cmd, "result": result})
            else:
                await websocket.send_json({"ack": cmd})
        except Exception as exc:
            logger.exception("Command %s failed", cmd)
            await websocket.send_json({"error": str(exc)})

    # ------------------------------------------------------------------
    # Status building
    # ------------------------------------------------------------------

    def _build_status(self) -> dict:
        """Build the current status snapshot."""
        recording = self.recording_ref.get("recording", False)
        episode = self.recording_ref.get("episode", 0)

        elapsed = 0.0
        if recording and self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time

        mode_val = self.state_ref.get("mode", "")
        if hasattr(mode_val, "value"):
            mode_val = mode_val.value

        ctrl_val = self.state_ref.get("control_mode", "")
        if hasattr(ctrl_val, "value"):
            ctrl_val = ctrl_val.value

        return {
            "mode": mode_val,
            "control": ctrl_val,
            "recording": recording,
            "episode": episode,
            "frames": self.recording_ref.get("frames", 0),
            "elapsed": round(elapsed, 1),
            "inference_ok": self.state_ref.get("inference_ok", True),
            "cameras": self.camera_names,
            "history": list(self._history),
        }

    # ------------------------------------------------------------------
    # Public helpers for the main loop
    # ------------------------------------------------------------------

    def on_episode_saved(self, episode_index: int, frames: int, success: int) -> None:
        """Call this after an episode is saved to update the history."""
        self._history.append({
            "episode": episode_index,
            "frames": frames,
            "success": success,
        })

    def on_recording_started(self) -> None:
        """Call this when recording starts."""
        self._start_time = time.perf_counter()

    def on_recording_stopped(self) -> None:
        """Call this when recording stops."""
        self._start_time = None
