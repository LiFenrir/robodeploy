"""数据采集区：左侧边栏（状态/控制/URDF 配置）+ 右侧直播区（上主相机大画面，下次要相机与 URDF 横排）。"""

import threading

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .widgets.video_grid import CameraView, VideoGrid
from .workers import UrdfWorker

DEFAULT_URDF_PATH = "/home/kemove/INNOV/infra/robot_SDK/robot-arm-4340/urdf/urdf/urdf.urdf"


class RecordTab(QWidget):
    """采集会话面板。控制命令写 pending_ref，由控制循环统一执行（与 WebUI 同一入口）。"""

    def __init__(
        self,
        *,
        state_ref: dict,
        recording_ref: dict,
        stop_ref: dict,
        obs_lock: threading.Lock,
        pending_ref: dict,
        camera_names: list[str],
        state_keys: list[str],
        fps: int,
        bridge,
        deploy_mode: bool = False,
        urdf_path: str = DEFAULT_URDF_PATH,
        urdf_joint_indices: str = "7,8,9,10,11,12",
        urdf_joint_scale: float = 1.0,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._state_ref = state_ref
        self._recording_ref = recording_ref
        self._stop_ref = stop_ref
        self._obs_lock = obs_lock
        self._pending_ref = pending_ref
        self._camera_names = camera_names
        self._state_keys = state_keys
        self._fps = fps
        self._bridge = bridge
        self._main_camera = camera_names[0] if camera_names else None
        self._urdf_worker: UrdfWorker | None = None

        self._build_ui(urdf_path, urdf_joint_indices, urdf_joint_scale)
        if deploy_mode:
            # 部署模式：隐藏录制/保存/模式切换，仅保留归零与退出
            self.btn_record.hide()
            self.btn_save.hide()
            self.btn_switch.hide()
        self._connect_bridge()
        self._init_shortcuts()

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # 30Hz 拉取最新帧，丢帧策略不阻塞控制循环
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, urdf_path: str, urdf_joint_indices: str, urdf_joint_scale: float) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # 左侧：可滚动侧边栏（状态 / 控制 / 主相机 / URDF / 历史）
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)

        status_box = QGroupBox("状态")
        status_form = QFormLayout(status_box)
        self.lbl_mode = QLabel("-")
        self.lbl_control = QLabel("-")
        self.lbl_recording = QLabel("-")
        self.lbl_episode = QLabel("0")
        self.lbl_frames = QLabel("0")
        self.lbl_elapsed = QLabel("0.0s")
        self.lbl_inference = QLabel("-")
        status_form.addRow("模式", self.lbl_mode)
        status_form.addRow("控制", self.lbl_control)
        status_form.addRow("录制", self.lbl_recording)
        status_form.addRow("剧集", self.lbl_episode)
        status_form.addRow("帧数", self.lbl_frames)
        status_form.addRow("耗时", self.lbl_elapsed)
        status_form.addRow("推理", self.lbl_inference)
        side_layout.addWidget(status_box)

        btn_box = QGroupBox("控制")
        btn_layout = QVBoxLayout(btn_box)
        self.btn_record = QPushButton("R 录制开始/停止")
        self.btn_record.clicked.connect(lambda: self._send_cmd("toggle_record"))
        self.btn_save = QPushButton("S 保存剧集")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_switch = QPushButton("P 切换模式")
        self.btn_switch.clicked.connect(lambda: self._send_cmd("switch_mode"))
        self.btn_zero = QPushButton("Z 归零")
        self.btn_zero.clicked.connect(lambda: self._send_cmd("reset_zero"))
        self.btn_exit = QPushButton("Esc 退出")
        self.btn_exit.clicked.connect(self._on_exit)
        for btn in (self.btn_record, self.btn_save, self.btn_switch, self.btn_zero, self.btn_exit):
            btn.setMinimumHeight(36)
            btn_layout.addWidget(btn)
        side_layout.addWidget(btn_box)

        view_box = QGroupBox("主视角")
        view_form = QFormLayout(view_box)
        self.combo_main = QComboBox()
        self.combo_main.addItems(self._camera_names)
        self.combo_main.currentTextChanged.connect(self._on_main_changed)
        view_form.addRow("主相机", self.combo_main)
        side_layout.addWidget(view_box)

        urdf_box = QGroupBox("URDF 随动视图")
        urdf_form = QFormLayout(urdf_box)
        urdf_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.check_urdf = QCheckBox("启用")
        self.check_urdf.toggled.connect(self._on_urdf_toggled)
        self.edit_urdf_path = QLineEdit(urdf_path)
        self.edit_urdf_path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.edit_urdf_indices = QLineEdit(urdf_joint_indices)
        self.edit_urdf_scale = QLineEdit(str(urdf_joint_scale))
        self.edit_urdf_scale.setMaximumWidth(70)
        self.lbl_urdf_status = QLabel("未启用")
        self.lbl_urdf_status.setWordWrap(True)
        urdf_form.addRow(self.check_urdf)
        # 路径较长，独占一行避免撑宽侧边栏
        urdf_form.addRow("URDF 路径", QLabel(""))
        urdf_form.addRow(self.edit_urdf_path)
        urdf_form.addRow("关节维度", self.edit_urdf_indices)
        urdf_form.addRow("弧度系数", self.edit_urdf_scale)
        urdf_form.addRow(self.lbl_urdf_status)
        side_layout.addWidget(urdf_box)

        history_box = QGroupBox("剧集历史")
        history_layout = QVBoxLayout(history_box)
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(120)
        history_layout.addWidget(self.history_list)
        side_layout.addWidget(history_box)
        side_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(420)
        splitter.addWidget(scroll)

        # 右侧：上下拆分 — 上主相机大画面，下次要相机 + URDF 横排
        live_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_view = CameraView(self._main_camera or "main")
        live_splitter.addWidget(self.main_view)
        self.video_grid = VideoGrid(single_row=True)
        for name in self._camera_names:
            if name != self._main_camera:
                self.video_grid.add_view(name)
        live_splitter.addWidget(self.video_grid)
        live_splitter.setStretchFactor(0, 3)
        live_splitter.setStretchFactor(1, 2)
        splitter.addWidget(live_splitter)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 960])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _connect_bridge(self) -> None:
        self._bridge.episode_saved.connect(self._on_episode_saved)
        self._bridge.label_requested.connect(self._on_label_requested)
        self._bridge.recording_started.connect(lambda: None)
        self._bridge.recording_stopped.connect(lambda: None)

    def _init_shortcuts(self) -> None:
        for key, handler in (
            ("R", lambda: self._send_cmd("toggle_record")),
            ("S", self._on_save),
            ("P", lambda: self._send_cmd("switch_mode")),
            ("Z", lambda: self._send_cmd("reset_zero")),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(handler)
        esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        esc.activated.connect(self._on_exit)

    # ------------------------------------------------------------------
    # 命令（写 pending_ref，与 WebUI 同一入口）
    # ------------------------------------------------------------------

    def _send_cmd(self, cmd: str, data: dict | None = None) -> None:
        self._pending_ref["cmd"] = cmd
        self._pending_ref["data"] = data

    def _ask_label(self) -> int | None:
        """弹出标注对话框。返回 1 成功 / 0 失败 / -1 丢弃，取消返回 None。"""
        box = QMessageBox(self)
        box.setWindowTitle("标记剧集")
        btn_ok = box.addButton("✓ 成功", QMessageBox.ButtonRole.AcceptRole)
        btn_fail = box.addButton("✗ 失败", QMessageBox.ButtonRole.DestructiveRole)
        btn_discard = box.addButton("🗑 丢弃", QMessageBox.ButtonRole.ActionRole)
        box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is btn_ok:
            return 1
        if clicked is btn_fail:
            return 0
        if clicked is btn_discard:
            return -1
        return None

    def _on_save(self) -> None:
        label = self._ask_label()
        if label is not None:
            self._send_cmd("save", {"label": label})

    def _on_exit(self) -> None:
        self._stop_ref["stop"] = True
        self.window().close()

    # ------------------------------------------------------------------
    # Bridge 信号
    # ------------------------------------------------------------------

    def _on_episode_saved(self, episode: int, frames: int, success: int) -> None:
        tag = {1: "✓", 0: "✗"}.get(success, "?")
        self.history_list.insertItem(0, f"{tag} #{episode} {frames}帧")

    def _on_label_requested(self) -> None:
        label = self._ask_label()
        self._bridge.set_label(label if label is not None else -1)

    # ------------------------------------------------------------------
    # 主视角切换
    # ------------------------------------------------------------------

    def _on_main_changed(self, name: str) -> None:
        old = self._main_camera
        if not name or name == old:
            return
        self._main_camera = name
        self.main_view.set_name(name)
        self.video_grid.remove_view(name)
        if old:
            self.video_grid.add_view(old)

    # ------------------------------------------------------------------
    # 30Hz 轮询：视频帧 + 状态 + URDF 关节角
    # ------------------------------------------------------------------

    def _poll(self) -> None:
        with self._obs_lock:
            obs = self._state_ref.get("obs")

        if obs:
            for cam in self._camera_names:
                frame = obs.get(cam)
                if frame is None:
                    continue
                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                frame = np.ascontiguousarray(frame[..., :3])
                if cam == self._main_camera:
                    self.main_view.set_frame(frame)
                else:
                    self.video_grid.set_frame(cam, frame)

            self._poll_urdf(obs)

        self._poll_status()

    def _poll_status(self) -> None:
        mode = self._state_ref.get("mode", "")
        control = self._state_ref.get("control_mode", "")
        self.lbl_mode.setText(getattr(mode, "value", str(mode)).upper())
        self.lbl_control.setText(getattr(control, "value", str(control)).upper())

        recording = self._recording_ref.get("recording", False)
        self.lbl_recording.setText("● REC" if recording else "○ 停止")
        self.lbl_recording.setStyleSheet("color: #ef4444;" if recording else "color: #888;")

        frames = self._recording_ref.get("frames", 0)
        self.lbl_episode.setText(str(self._recording_ref.get("episode", 0)))
        self.lbl_frames.setText(str(frames))
        self.lbl_elapsed.setText(f"{frames / self._fps:.1f}s" if recording else "0.0s")

        inference_ok = self._state_ref.get("inference_ok", True)
        self.lbl_inference.setText("OK" if inference_ok else "ERR")
        self.lbl_inference.setStyleSheet("color: #16a34a;" if inference_ok else "color: #ef4444;")

    # ------------------------------------------------------------------
    # URDF 随动
    # ------------------------------------------------------------------

    def _on_urdf_toggled(self, checked: bool) -> None:
        if checked:
            path = self.edit_urdf_path.text().strip()
            self.video_grid.add_view("urdf")
            self._urdf_worker = UrdfWorker(path, parent=self)
            self._urdf_worker.ready.connect(lambda nq: self.lbl_urdf_status.setText(f"渲染中 (nq={nq})"))
            self._urdf_worker.failed.connect(self._on_urdf_failed)
            self._urdf_worker.frame_ready.connect(lambda f: self.video_grid.set_frame("urdf", f))
            self._urdf_worker.start()
        else:
            self._stop_urdf()

    def _stop_urdf(self) -> None:
        if self._urdf_worker is not None:
            self._urdf_worker.stop()
            self._urdf_worker = None
        self.video_grid.remove_view("urdf")
        self.lbl_urdf_status.setText("未启用")

    def _on_urdf_failed(self, message: str) -> None:
        # 渲染上下文初始化失败 → 降级为关节角数值显示
        self._urdf_worker = None
        self.video_grid.remove_view("urdf")
        self.lbl_urdf_status.setText(f"{message}（已降级为数值显示）")

    def _poll_urdf(self, obs: dict) -> None:
        if not self.check_urdf.isChecked():
            return
        try:
            indices = [int(s) for s in self.edit_urdf_indices.text().split(",") if s.strip()]
            scale = float(self.edit_urdf_scale.text())
            state = np.array([obs.get(k, 0.0) for k in self._state_keys], dtype=np.float64)
            values = state[indices] * scale
        except (ValueError, IndexError) as exc:
            self.lbl_urdf_status.setText(f"关节维度配置错误: {exc}")
            return
        if self._urdf_worker is not None:
            self._urdf_worker.set_qpos(values)
        else:
            self.lbl_urdf_status.setText("关节角: " + " ".join(f"{v:.3f}" for v in values))

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._timer.stop()
        self._stop_urdf()
