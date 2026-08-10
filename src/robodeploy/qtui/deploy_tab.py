"""部署专区：策略客户端配置 + 连接测试 + policy 模式控制循环启动（复用 Qt 采集前端）。"""

import socket
import sys

from PyQt6.QtCore import QProcess, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from robodeploy.policy_clients import make_policy_client_from_config
from robodeploy.policy_clients.lingbot.config import LingbotPolicyClientConfig
from robodeploy.policy_clients.openpi.config import OpenPIPolicyClientConfig

from .param_schemas import ROBOT_PARAMS, DynamicParamsForm
from .widgets.camera_slots import CameraSlotsEditor

ROBOT_TYPES = list(ROBOT_PARAMS)


class _ConnectWorker(QThread):
    """策略服务端连接测试：socket 预检 + make_policy_client + get_server_metadata。"""

    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg, host: str, port: int, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._host = host
        self._port = port

    def run(self) -> None:
        try:
            with socket.create_connection((self._host, self._port), timeout=3):
                pass
        except OSError as exc:
            self.failed.emit(f"无法连接 {self._host}:{self._port}（{exc}）")
            return
        try:
            client = make_policy_client_from_config(self._cfg)
            if not client.connected:
                self.failed.emit("客户端未连接")
                return
            metadata = client.get_server_metadata()
            self.done.emit(f"连接成功，服务端信息: {metadata}")
        except Exception as exc:
            self.failed.emit(f"连接失败: {exc}")


class DeployTab(QWidget):
    """部署面板：control_mode=policy + 不录制，启动 record_dataset --front_end=qt 子进程。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._connect_worker: _ConnectWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # 左侧：可滚动参数侧边栏（与采集区一致）
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)

        policy_box = QGroupBox("策略客户端")
        form = QFormLayout(policy_box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.combo_policy = QComboBox()
        self.combo_policy.addItems(["openpi", "lingbot"])
        self.edit_host = QLineEdit("localhost")
        self.edit_port = QLineEdit("8000")
        self.check_rtc = QCheckBox("启用 RTC")
        self.edit_horizon = QLineEdit("13")
        self.edit_robo_name = QLineEdit("bi_s1")
        self.edit_task = QLineEdit("fold the box")
        for widget in (self.combo_policy, self.edit_host, self.edit_robo_name, self.edit_task):
            widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        form.addRow("类型", self.combo_policy)
        form.addRow("host", self.edit_host)
        form.addRow("port", self.edit_port)
        form.addRow(self.check_rtc)
        form.addRow("rtc_execution_horizon", self.edit_horizon)
        form.addRow("robo_name（lingbot）", self.edit_robo_name)
        form.addRow("task（提示词）", self.edit_task)
        side_layout.addWidget(policy_box)

        smooth_box = QGroupBox("动作平滑（StreamBuffer，RTC 关闭时生效）")
        smooth_form = QFormLayout(smooth_box)
        self.check_smoothing = QCheckBox("启用 temporal smoothing")
        self.check_smoothing.setChecked(True)
        self.edit_inference_rate = QLineEdit("3.0")
        self.edit_latency_k = QLineEdit("8")
        self.edit_min_smooth = QLineEdit("8")
        self.edit_action_smooth = QLineEdit("0.05")
        smooth_form.addRow(self.check_smoothing)
        smooth_form.addRow("inference_rate (Hz)", self.edit_inference_rate)
        smooth_form.addRow("latency_k", self.edit_latency_k)
        smooth_form.addRow("min_smooth_steps", self.edit_min_smooth)
        smooth_form.addRow("action_smooth_max_step", self.edit_action_smooth)
        side_layout.addWidget(smooth_box)

        robot_box = QGroupBox("机器人")
        robot_form = QFormLayout(robot_box)
        robot_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.combo_robot = QComboBox()
        self.combo_robot.setEditable(True)
        self.combo_robot.addItems(ROBOT_TYPES)
        self.combo_robot.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.robot_params = DynamicParamsForm()
        self.combo_robot.currentTextChanged.connect(
            lambda t: self.robot_params.set_schema(ROBOT_PARAMS.get(t.strip(), []))
        )
        self.robot_params.set_schema(ROBOT_PARAMS.get(self.combo_robot.currentText(), []))
        robot_form.addRow("robot.type", self.combo_robot)
        robot_form.addRow(self.robot_params)
        side_layout.addWidget(robot_box)

        cam_box = QGroupBox("相机（与相机调试区导出 JSON 联动）")
        cam_layout = QVBoxLayout(cam_box)
        self.cam_editor = CameraSlotsEditor()
        cam_layout.addWidget(self.cam_editor)
        side_layout.addWidget(cam_box)
        side_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(380)
        splitter.addWidget(scroll)

        # 右侧：操作 + 状态 + 日志
        right = QWidget()
        right_layout = QVBoxLayout(right)
        btn_row = QHBoxLayout()
        self.btn_test = QPushButton("连接测试")
        self.btn_test.clicked.connect(self._on_test)
        self.btn_launch = QPushButton("启动部署")
        self.btn_launch.clicked.connect(self._on_launch)
        self.btn_stop = QPushButton("终止进程")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_test)
        btn_row.addWidget(self.btn_launch)
        btn_row.addWidget(self.btn_stop)
        right_layout.addLayout(btn_row)

        self.lbl_status = QLabel("未连接")
        self.lbl_status.setWordWrap(True)
        right_layout.addWidget(self.lbl_status)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        right_layout.addWidget(self.log_view)
        splitter.addWidget(right)

        splitter.setSizes([340, 940])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _policy_config(self):
        host = self.edit_host.text().strip()
        port = int(self.edit_port.text())
        if self.combo_policy.currentText() == "openpi":
            return OpenPIPolicyClientConfig(
                host=host,
                port=port,
                use_rtc=self.check_rtc.isChecked(),
                rtc_execution_horizon=int(self.edit_horizon.text()),
            )
        return LingbotPolicyClientConfig(host=host, port=port, robo_name=self.edit_robo_name.text().strip())

    def _on_test(self) -> None:
        if self._connect_worker is not None and self._connect_worker.isRunning():
            return
        self.lbl_status.setText("连接中…")
        self.btn_test.setEnabled(False)
        self._connect_worker = _ConnectWorker(
            self._policy_config(), self.edit_host.text().strip(), int(self.edit_port.text()), self
        )
        self._connect_worker.done.connect(self._on_test_done)
        self._connect_worker.failed.connect(self._on_test_done)
        self._connect_worker.start()

    def _on_test_done(self, message: str) -> None:
        self.lbl_status.setText(message)
        self.btn_test.setEnabled(True)

    def _on_launch(self) -> None:
        if self._process is not None:
            return
        args = [
            "-m",
            "robodeploy.scripts.record_dataset",
            "--front_end=qt",
            "--control_mode=policy",
            "--deploy_mode=true",
            "--robot.type",
            self.combo_robot.currentText(),
            "--task",
            self.edit_task.text(),
        ]
        for key, value in self.robot_params.values().items():
            args += [f"--robot.{key}", value]
        cameras_json = self.cam_editor.to_json()
        if cameras_json != "{}":
            args += ["--robot.cameras", cameras_json]
        args += [
            "--use_temporal_smoothing",
            "true" if self.check_smoothing.isChecked() else "false",
            "--inference_rate",
            self.edit_inference_rate.text(),
            "--latency_k",
            self.edit_latency_k.text(),
            "--min_smooth_steps",
            self.edit_min_smooth.text(),
            "--action_smooth_max_step",
            self.edit_action_smooth.text(),
        ]
        if self.combo_policy.currentText() == "openpi":
            args += [
                "--policy.type=openpi",
                f"--policy.host={self.edit_host.text().strip()}",
                f"--policy.port={self.edit_port.text()}",
            ]
            args += ["--use_rtc", "true" if self.check_rtc.isChecked() else "false"]
            args += ["--rtc_execution_horizon", self.edit_horizon.text()]
        else:
            args += [
                "--policy.type=lingbot",
                f"--policy.host={self.edit_host.text().strip()}",
                f"--policy.port={self.edit_port.text()}",
                f"--policy.robo_name={self.edit_robo_name.text().strip()}",
            ]

        self.log_view.appendPlainText(f"$ {sys.executable} {' '.join(args)}\n")
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_finished)
        self._process.start(sys.executable, args)
        self.btn_launch.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _on_output(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log_view.appendPlainText(data.rstrip())

    def _on_finished(self) -> None:
        self.log_view.appendPlainText("\n[进程已退出]")
        self._process = None
        self.btn_launch.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_stop(self) -> None:
        if self._process is not None:
            self._process.kill()

    def shutdown(self) -> None:
        self._on_stop()
