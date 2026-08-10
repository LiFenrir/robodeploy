"""Qt 采集前端挂接：主线程 QApplication，控制循环在 worker 线程，共享 state_ref/pending_ref。"""

import logging
import os
import sys
import threading
from collections.abc import Callable

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from .app import MainWindow
from .record_tab import RecordTab
from .theme import apply_theme

logger = logging.getLogger(__name__)


class QtRecordBridge(QObject):
    """record_loop 钩子桥：与 WebUIServer 相同的 on_* 接口，经 Qt 信号转发到主线程。

    request_label 由 worker 线程调用，阻塞至用户在对话框完成选择。
    """

    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    episode_saved = pyqtSignal(int, int, int)
    label_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._label_event = threading.Event()
        self._label_result = -1

    # WebUIServer 兼容钩子
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def on_recording_started(self) -> None:
        self.recording_started.emit()

    def on_recording_stopped(self) -> None:
        self.recording_stopped.emit()

    def on_episode_saved(self, episode_index: int, frames: int, success: int) -> None:
        self.episode_saved.emit(episode_index, frames, success)

    def request_label(self) -> int:
        self._label_event.clear()
        self.label_requested.emit()
        self._label_event.wait()
        return self._label_result

    def set_label(self, label: int) -> None:
        self._label_result = label
        self._label_event.set()


def run_qt_session(
    *,
    bridge: QtRecordBridge,
    loop_fn: Callable[[], None],
    state_ref: dict,
    recording_ref: dict,
    stop_ref: dict,
    obs_lock: threading.Lock,
    pending_ref: dict,
    camera_names: list[str],
    state_keys: list[str],
    fps: int,
    urdf_path: str,
    urdf_joint_indices: str,
    urdf_joint_scale: float,
    deploy_mode: bool = False,
) -> None:
    """启动 Qt 采集会话（阻塞至窗口关闭）。loop_fn 在后台线程运行控制循环。"""
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)

    record_tab = RecordTab(
        state_ref=state_ref,
        recording_ref=recording_ref,
        stop_ref=stop_ref,
        obs_lock=obs_lock,
        pending_ref=pending_ref,
        camera_names=camera_names,
        state_keys=state_keys,
        fps=fps,
        bridge=bridge,
        deploy_mode=deploy_mode,
        urdf_path=urdf_path,
        urdf_joint_indices=urdf_joint_indices,
        urdf_joint_scale=urdf_joint_scale,
    )
    win = MainWindow(record_tab=record_tab)

    def _loop() -> None:
        try:
            loop_fn()
        except Exception:
            logger.exception("控制循环异常退出")
        finally:
            stop_ref["stop"] = True

    loop_thread = threading.Thread(target=_loop, daemon=True, name="record_loop")
    loop_thread.start()

    # 控制循环退出（异常/停止）时自动关窗
    watchdog = QTimer(win)
    watchdog.setInterval(500)
    watchdog.timeout.connect(lambda: win.close() if not loop_thread.is_alive() else None)
    watchdog.start()

    # 窗口关闭（含点叉）立即通知控制循环退出，不必等 exec 返回
    app.aboutToQuit.connect(lambda: stop_ref.__setitem__("stop", True))

    win.show()
    app.exec()

    stop_ref["stop"] = True
    record_tab.shutdown()
    loop_thread.join(timeout=10)
    if loop_thread.is_alive():
        # 控制循环卡在硬件 IO 等位置无法干净退出 → 强制结束，避免终端挂住
        print("控制循环未在 10s 内退出，强制结束进程", file=sys.stderr, flush=True)
        os._exit(0)
