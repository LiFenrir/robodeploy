"""相机槽位编辑器：采集/部署共用。生成 --robot.cameras JSON，支持加载调试区导出文件。"""

import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

CAMERA_TYPES = ["intelrealsense", "opencv"]
DEFAULT_CAMERAS = ["front", "left_wrist", "right_wrist"]


class CameraSlot(QWidget):
    """单路相机配置（两行）：上行 名称/类型/id，下行 宽/高/fps/删除。"""

    def __init__(self, name: str = "", parent: QWidget | None = None, on_remove=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        row1 = QHBoxLayout()
        self.edit_name = QLineEdit(name)
        self.edit_name.setPlaceholderText("名称")
        self.edit_name.setMaximumWidth(64)
        self.combo_type = QComboBox()
        self.combo_type.addItems(CAMERA_TYPES)
        self.combo_type.setMaximumWidth(112)
        self.edit_id = QLineEdit()
        self.edit_id.setPlaceholderText("序列号 / /dev/videoN")
        row1.addWidget(self.edit_name)
        row1.addWidget(self.combo_type)
        row1.addWidget(self.edit_id, 1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.edit_width = QLineEdit("640")
        self.edit_height = QLineEdit("480")
        self.edit_fps = QLineEdit("30")
        for label, edit in (("宽", self.edit_width), ("高", self.edit_height), ("fps", self.edit_fps)):
            row2.addWidget(QLabel(label))
            row2.addWidget(edit)
        row2.addStretch(1)
        btn_remove = QPushButton("×")
        btn_remove.setObjectName("slotRemove")
        btn_remove.setMaximumWidth(28)
        if on_remove is not None:
            btn_remove.clicked.connect(on_remove)
        row2.addWidget(btn_remove)
        layout.addLayout(row2)

    def to_config(self) -> tuple[str, dict] | None:
        """转 --robot.cameras 条目；名称或 id 为空返回 None。"""
        name = self.edit_name.text().strip()
        cam_id = self.edit_id.text().strip()
        if not name or not cam_id:
            return None
        entry: dict = {"type": self.combo_type.currentText()}
        if entry["type"] == "opencv":
            entry["index_or_path"] = int(cam_id) if cam_id.isdigit() else cam_id
        else:
            entry["serial_number_or_name"] = cam_id
        entry["width"] = int(self.edit_width.text())
        entry["height"] = int(self.edit_height.text())
        entry["fps"] = int(self.edit_fps.text())
        return name, entry

    def load_config(self, name: str, entry: dict) -> None:
        self.edit_name.setText(name)
        cam_type = entry.get("type", "intelrealsense")
        self.combo_type.setCurrentText(cam_type if cam_type in CAMERA_TYPES else "intelrealsense")
        cam_id = entry.get("serial_number_or_name", entry.get("index_or_path", ""))
        self.edit_id.setText(str(cam_id))
        self.edit_width.setText(str(entry.get("width", 640)))
        self.edit_height.setText(str(entry.get("height", 480)))
        self.edit_fps.setText(str(entry.get("fps", 30)))


class CameraSlotsEditor(QWidget):
    """相机槽位列表 + 添加/加载按钮。to_json() 生成 --robot.cameras 值。"""

    def __init__(self, default_names: list[str] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._slots: list[CameraSlot] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._slots_layout = QVBoxLayout()
        self._slots_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._slots_layout)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("+ 添加相机")
        btn_add.clicked.connect(lambda: self.add_slot(""))
        btn_load = QPushButton("加载相机参数…")
        btn_load.clicked.connect(self.load_from_file)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_load)
        layout.addLayout(btn_row)

        for name in default_names if default_names is not None else DEFAULT_CAMERAS:
            self.add_slot(name)

    def add_slot(self, name: str) -> CameraSlot:
        slot = CameraSlot(name, on_remove=lambda: self.remove_slot(slot))
        self._slots.append(slot)
        self._slots_layout.addWidget(slot)
        return slot

    def remove_slot(self, slot: CameraSlot) -> None:
        self._slots.remove(slot)
        self._slots_layout.removeWidget(slot)
        slot.deleteLater()

    def load_from_file(self) -> bool:
        path, _ = QFileDialog.getOpenFileName(self, "加载相机参数", "", "JSON (*.json)")
        if not path:
            return False
        cameras = json.loads(Path(path).read_text(encoding="utf-8"))
        for slot in list(self._slots):
            self.remove_slot(slot)
        for name, entry in cameras.items():
            self.add_slot(name).load_config(name, entry)
        return True

    def to_json(self) -> str:
        """生成 --robot.cameras JSON；无有效槽位返回 "{}"。"""
        cameras = {}
        for slot in self._slots:
            item = slot.to_config()
            if item is not None:
                cameras[item[0]] = item[1]
        return json.dumps(cameras)
