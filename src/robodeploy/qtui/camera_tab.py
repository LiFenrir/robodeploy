"""相机调试区：枚举 / 参数配置 / 实时预览 / 内参导出 / 采集参数导出（供数据采集区加载）。

预览取流走底层 cv2 / pyrealsense2 API（workers.CameraWorker），参数尽力设置，
状态栏显示相机实际生效的流参数。
"""

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from robodeploy.cameras.configs import Cv2Rotation

from .widgets.video_grid import VideoGrid
from .workers import CameraWorker, EnumWorker, res_pixels

# OpenCV 等无档位枚举能力时的常用预设（下拉框可编辑，允许手输其他值）
_DEFAULT_RESOLUTIONS = ["640x480", "848x480", "1280x720", "1920x1080"]
_DEFAULT_FPS = ["6", "15", "30", "60"]


class CameraTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._enum_worker: EnumWorker | None = None
        self._workers: dict[str, CameraWorker] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        enum_box = QGroupBox("相机列表")
        enum_layout = QVBoxLayout(enum_box)
        self.btn_enum = QPushButton("枚举相机")
        self.btn_enum.clicked.connect(self._on_enumerate)
        self.cam_list = QListWidget()
        self.cam_list.currentItemChanged.connect(self._on_camera_selected)
        enum_layout.addWidget(self.btn_enum)
        enum_layout.addWidget(self.cam_list)
        left_layout.addWidget(enum_box)

        param_box = QGroupBox("参数配置（连接后修改，点应用生效）")
        form = QFormLayout(param_box)
        self.combo_resolution = QComboBox()
        self.combo_resolution.setEditable(True)
        self.combo_resolution.addItems(_DEFAULT_RESOLUTIONS)
        self.combo_fps = QComboBox()
        self.combo_fps.setEditable(True)
        self.combo_fps.addItems(_DEFAULT_FPS)
        self.combo_fps.setCurrentText("30")
        self.combo_resolution.currentTextChanged.connect(lambda _t: self._fill_fps_combo())
        self.combo_rotation = QComboBox()
        for r in Cv2Rotation:
            self.combo_rotation.addItem(str(r.value), r)
        self.check_depth = QCheckBox("启用深度流（仅 RealSense）")
        form.addRow("分辨率", self.combo_resolution)
        form.addRow("fps", self.combo_fps)
        form.addRow("rotation", self.combo_rotation)
        form.addRow(self.check_depth)
        left_layout.addWidget(param_box)

        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("连接预览 / 应用参数")
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect = QPushButton("断开")
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_disconnect)
        left_layout.addLayout(btn_row)

        export_row = QHBoxLayout()
        self.btn_export = QPushButton("导出内参 JSON")
        self.btn_export.clicked.connect(self._on_export_intrinsics)
        self.btn_export_cfg = QPushButton("导出采集参数 JSON")
        self.btn_export_cfg.clicked.connect(self._on_export_record_config)
        export_row.addWidget(self.btn_export)
        export_row.addWidget(self.btn_export_cfg)
        left_layout.addLayout(export_row)

        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        self.video_grid = VideoGrid()
        splitter.addWidget(left)
        splitter.addWidget(self.video_grid)
        splitter.setSizes([320, 960])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # 枚举
    # ------------------------------------------------------------------

    def _on_enumerate(self) -> None:
        if self._enum_worker is not None and self._enum_worker.isRunning():
            return
        self.btn_enum.setEnabled(False)
        self.status_label.setText("枚举中…")
        self._enum_worker = EnumWorker(self)
        self._enum_worker.cameras_found.connect(self._on_cameras_found)
        self._enum_worker.start()

    def _on_cameras_found(self, cameras: list) -> None:
        self.btn_enum.setEnabled(True)
        self.cam_list.clear()
        for info in cameras:
            if info.get("type") == "error":
                item = QListWidgetItem(info["name"])
                item.setFlags(Qt.ItemFlag.NoItemFlags)
            else:
                profile = info.get("default_stream_profile") or {}
                text = f"[{info['type']}] {info['name']} | {profile.get('width', '?')}x{profile.get('height', '?')}@{profile.get('fps', '?')}"
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, info)
            self.cam_list.addItem(item)
        self.status_label.setText(f"发现 {self.cam_list.count()} 台设备")

    # ------------------------------------------------------------------
    # 连接 / 断开 / 动态应用参数
    # ------------------------------------------------------------------

    def _selected_camera_info(self) -> dict | None:
        item = self.cam_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _on_camera_selected(self, item, _prev) -> None:
        """选中相机后按其真实档位填充分辨率/fps 下拉（无档位信息时用预设，仍可手输）。"""
        info = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        modes = (info or {}).get("supported_modes") or {}
        # Qt item data 往返后 dict 顺序可能被打乱，这里重新按像素数降序
        resolutions = sorted(modes, key=lambda r: -res_pixels(r)) if modes else list(_DEFAULT_RESOLUTIONS)
        current = self.combo_resolution.currentText()
        self.combo_resolution.blockSignals(True)
        self.combo_resolution.clear()
        self.combo_resolution.addItems(resolutions)
        if current in resolutions:
            self.combo_resolution.setCurrentText(current)
        self.combo_resolution.blockSignals(False)
        self._fill_fps_combo()

    def _fill_fps_combo(self) -> None:
        modes = (self._selected_camera_info() or {}).get("supported_modes") or {}
        fps_list = [str(f) for f in modes.get(self.combo_resolution.currentText(), [])] or list(_DEFAULT_FPS)
        current = self.combo_fps.currentText()
        self.combo_fps.blockSignals(True)
        self.combo_fps.clear()
        self.combo_fps.addItems(fps_list)
        if current in fps_list:
            self.combo_fps.setCurrentText(current)
        else:
            # 当前 fps 在该分辨率下无效 → 吸附到数值最接近的有效档
            try:
                nearest = min(fps_list, key=lambda f: abs(float(f) - float(current)))
            except ValueError:
                nearest = "30" if "30" in fps_list else fps_list[0]
            self.combo_fps.setCurrentText(nearest)
        self.combo_fps.blockSignals(False)

    def _build_source(self, info: dict) -> dict:
        """由表单参数构建取流源描述（传给底层 CameraWorker）。"""
        try:
            w_str, h_str = self.combo_resolution.currentText().lower().replace("×", "x").split("x")
            width, height = int(w_str), int(h_str)
            fps = round(float(self.combo_fps.currentText()))
        except ValueError:
            raise ValueError(
                f"分辨率/fps 格式错误: {self.combo_resolution.currentText()} @ {self.combo_fps.currentText()}"
            ) from None
        source = {
            "width": width,
            "height": height,
            "fps": fps,
            "rotation": int(self.combo_rotation.currentData().value),
        }
        if info["type"] == "OpenCV":
            cam_id = info["id"]
            if isinstance(cam_id, str) and cam_id.isdigit():
                cam_id = int(cam_id)
            source.update({"kind": "opencv", "id": cam_id})
        elif info["type"] == "RealSense":
            source.update(
                {"kind": "realsense", "serial": str(info["id"]), "use_depth": self.check_depth.isChecked()}
            )
        else:
            raise ValueError(f"未知相机类型: {info['type']}")
        return source

    def _on_connect(self) -> None:
        info = self._selected_camera_info()
        if info is None:
            QMessageBox.warning(self, "提示", "请先枚举并选择一台相机")
            return

        try:
            source = self._build_source(info)
        except ValueError as exc:
            QMessageBox.critical(self, "配置错误", str(exc))
            return

        view_name = str(info["id"])
        # 已连接 → 停止旧取流，用新参数重连（动态应用）
        old = self._workers.pop(view_name, None)
        if old is not None:
            old.stop()
            self.video_grid.remove_view(view_name)

        worker = CameraWorker(source, self)
        worker.opened.connect(lambda actual, n=view_name: self._on_opened(n, actual))
        worker.frame_ready.connect(lambda frame, n=view_name: self.video_grid.set_frame(n, frame))
        worker.failed.connect(lambda msg, n=view_name: self._on_failed(n, msg))
        self._workers[view_name] = worker
        worker.start()
        self.status_label.setText(f"连接中: {view_name}")

    def _on_opened(self, name: str, actual: str) -> None:
        self.video_grid.add_view(name)
        self.status_label.setText(f"已连接: {name}（实际流 {actual}）")

    def _on_failed(self, name: str, message: str) -> None:
        worker = self._workers.pop(name, None)
        if worker is not None:
            worker.stop()
        self.video_grid.remove_view(name)
        self.status_label.setText(message)

    def _on_disconnect(self) -> None:
        info = self._selected_camera_info()
        if info is None:
            return
        name = str(info["id"])
        worker = self._workers.pop(name, None)
        if worker is not None:
            worker.stop()
            self.video_grid.remove_view(name)
            self.status_label.setText(f"已断开: {name}")

    # ------------------------------------------------------------------
    # 内参导出
    # ------------------------------------------------------------------

    def _on_export_intrinsics(self) -> None:
        info = self._selected_camera_info()
        if info is None:
            QMessageBox.warning(self, "提示", "请先选择相机")
            return
        name = str(info["id"])
        worker = self._workers.get(name)
        if worker is None or not worker.isRunning():
            QMessageBox.warning(self, "提示", "请先连接预览该相机")
            return
        if worker.intrinsics is None:
            QMessageBox.warning(self, "提示", "仅 RealSense 相机支持内参读取")
            return

        payload = {"camera_id": name, "stream": worker.actual, "intrinsics": worker.intrinsics}
        path, _ = QFileDialog.getSaveFileName(self, "导出内参", f"intrinsics_{name}.json", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self.status_label.setText(f"内参已导出: {path}")

    # ------------------------------------------------------------------
    # 采集参数导出（--robot.cameras JSON，供数据采集区加载）
    # ------------------------------------------------------------------

    def _record_config_entry(self, worker: CameraWorker) -> dict:
        """单台相机的采集配置：用实际生效的流参数。"""
        source = worker.source
        if source["kind"] == "realsense":
            entry = {"type": "intelrealsense", "serial_number_or_name": source["serial"]}
        else:
            entry = {"type": "opencv", "index_or_path": source["id"]}
        entry["width"] = worker.actual["width"]
        entry["height"] = worker.actual["height"]
        entry["fps"] = round(worker.actual["fps"])
        if source.get("rotation"):
            entry["rotation"] = source["rotation"]
        return entry

    def _on_export_record_config(self) -> None:
        cameras = {name: self._record_config_entry(w) for name, w in self._workers.items() if w.isRunning()}
        if not cameras:
            QMessageBox.warning(self, "提示", "没有已连接的相机")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出采集参数", "cameras.json", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(cameras, indent=2, default=str), encoding="utf-8")
        self.status_label.setText(f"采集参数已导出: {path}（{len(cameras)} 台相机）")
