"""采集启动配置页：生成 record_dataset CLI 参数，QProcess 子进程启动（--front_end=qt）。

相机配置与相机调试区联动：调试区导出 cameras.json，此处一键加载；
默认 3 路相机槽位，"+" 按钮添加额外相机。
"""

import sys
from datetime import datetime

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
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

from .param_schemas import ROBOT_PARAMS, TELEOP_PARAMS, DynamicParamsForm
from .widgets.camera_slots import CameraSlotsEditor

ROBOT_TYPES = list(ROBOT_PARAMS)
TELEOP_TYPES = ["", *TELEOP_PARAMS]


class RecordLauncherTab(QWidget):
    """RecordConfig 表单 + 子进程启动 record_dataset --front_end=qt，日志实时显示。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # 左侧：可滚动参数侧边栏
        sidebar = QWidget()
        form = QFormLayout(sidebar)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.combo_robot = QComboBox()
        self.combo_robot.setEditable(True)
        self.combo_robot.addItems(ROBOT_TYPES)
        self.combo_teleop = QComboBox()
        self.combo_teleop.setEditable(True)
        self.combo_teleop.addItems(TELEOP_TYPES)
        self.edit_task = QLineEdit("fold the box")
        self.edit_repo_id = QLineEdit("dataset")
        self.edit_output_dir = QLineEdit()
        self.edit_output_dir.setPlaceholderText("留空使用默认路径")
        self.btn_output_dir = QPushButton("浏览…")
        self.btn_output_dir.clicked.connect(self._on_browse_output_dir)
        self.lbl_output_default = QLabel("")
        self.lbl_output_default.setStyleSheet("color: #888;")
        self.lbl_output_default.setWordWrap(True)
        self.edit_fps = QLineEdit("30")
        self.edit_episode_time = QLineEdit("120")
        # 长字段允许收缩省略，全部字段统一全宽对齐
        shrinkable = (
            self.combo_robot,
            self.combo_teleop,
            self.edit_task,
            self.edit_repo_id,
            self.edit_output_dir,
        )
        for widget in shrinkable:
            widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        # robot/teleop 子参数随类型动态加载
        self.robot_params = DynamicParamsForm()
        self.teleop_params = DynamicParamsForm()
        self.combo_robot.currentTextChanged.connect(self._on_robot_type_changed)
        self.combo_teleop.currentTextChanged.connect(self._on_teleop_type_changed)

        form.addRow("robot.type", self.combo_robot)
        form.addRow(self.robot_params)
        form.addRow("teleop.type", self.combo_teleop)
        form.addRow(self.teleop_params)
        form.addRow("task", self.edit_task)
        form.addRow("repo_id", self.edit_repo_id)
        output_row = QHBoxLayout()
        output_row.addWidget(self.edit_output_dir, 1)
        output_row.addWidget(self.btn_output_dir)
        form.addRow("output_dir", output_row)
        form.addRow("", self.lbl_output_default)
        form.addRow("fps", self.edit_fps)
        form.addRow("episode_time_s", self.edit_episode_time)

        cam_box = QGroupBox("相机（与相机调试区导出 JSON 联动）")
        cam_layout = QVBoxLayout(cam_box)
        self.cam_editor = CameraSlotsEditor()
        cam_layout.addWidget(self.cam_editor)
        form.addRow(cam_box)

        self._on_robot_type_changed(self.combo_robot.currentText())
        self._on_teleop_type_changed(self.combo_teleop.currentText())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(380)
        splitter.addWidget(scroll)

        # 右侧：启动按钮 + 日志
        right = QWidget()
        right_layout = QVBoxLayout(right)
        btn_row = QHBoxLayout()
        self.btn_launch = QPushButton("启动采集（Qt 前端）")
        self.btn_launch.clicked.connect(self._on_launch)
        self.btn_stop = QPushButton("终止进程")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self.btn_launch)
        btn_row.addWidget(self.btn_stop)
        right_layout.addLayout(btn_row)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        right_layout.addWidget(self.log_view)
        splitter.addWidget(right)

        splitter.setSizes([340, 940])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # 动态子参数 / output_dir
    # ------------------------------------------------------------------

    def _on_robot_type_changed(self, robot_type: str) -> None:
        self.robot_params.set_schema(ROBOT_PARAMS.get(robot_type.strip(), []))
        self._update_output_default()

    def _on_teleop_type_changed(self, teleop_type: str) -> None:
        self.teleop_params.set_schema(TELEOP_PARAMS.get(teleop_type.strip(), []))

    def _on_browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.edit_output_dir.setText(path)
            self._update_output_default()

    def _update_output_default(self) -> None:
        # 与 record_dataset.py 的 auto 解析一致：outputs/<robot>/<MMDD_HHMM>
        text = self.edit_output_dir.text().strip()
        if text:
            self.lbl_output_default.setText("")
        else:
            stamp = datetime.now().strftime("%m%d_%H%M")
            robot = self.combo_robot.currentText().strip() or "robot"
            self.lbl_output_default.setText(f"默认: outputs/{robot}/{stamp}（启动时按当时时间生成）")

    # ------------------------------------------------------------------
    # 启动
    # ------------------------------------------------------------------

    def _build_args(self) -> list[str]:
        args = ["-m", "robodeploy.scripts.record_dataset", "--front_end=qt"]
        args += ["--robot.type", self.combo_robot.currentText()]
        for key, value in self.robot_params.values().items():
            args += [f"--robot.{key}", value]
        if self.combo_teleop.currentText():
            args += ["--teleop.type", self.combo_teleop.currentText()]
            for key, value in self.teleop_params.values().items():
                args += [f"--teleop.{key}", value]
        cameras_json = self.cam_editor.to_json()
        if cameras_json != "{}":
            args += ["--robot.cameras", cameras_json]
        args += ["--task", self.edit_task.text()]
        args += ["--repo_id", self.edit_repo_id.text()]
        output_dir = self.edit_output_dir.text().strip()
        if output_dir:
            args += ["--output_dir", output_dir]
        args += ["--fps", self.edit_fps.text()]
        args += ["--episode_time_s", self.edit_episode_time.text()]
        return args

    def _on_launch(self) -> None:
        if self._process is not None:
            return
        try:
            args = self._build_args()
        except ValueError as exc:
            self.log_view.appendPlainText(f"参数错误（相机宽高/fps 须为整数）: {exc}")
            return
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
