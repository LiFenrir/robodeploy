"""按 robot/teleop 类型动态生成的子参数表单。切换类型时同名参数保留已填值。"""

from PyQt6.QtWidgets import QFormLayout, QLineEdit, QSizePolicy, QWidget

# 类型 → [(参数名, 默认值)]，与各 Config dataclass 字段对应（cameras 除外，单独配置）
ROBOT_PARAMS: dict[str, list[tuple[str, str]]] = {
    "bi_innov_arm_v1": [
        ("left_port", "/dev/ttyACM1"),
        ("right_port", "/dev/ttyACM0"),
        ("mode", "collect"),
    ],
    "innov_arm_v1": [("port", "/dev/ttyACM0"), ("mode", "collect")],
    "s1_follower": [("port", "/dev/ttyACM0")],
    "bi_s1_follower": [
        ("left_arm_port", "/dev/ttyACM0"),
        ("right_arm_port", "/dev/ttyACM1"),
    ],
}

TELEOP_PARAMS: dict[str, list[tuple[str, str]]] = {
    "s1_leader": [("port", "/dev/ttyACM2")],
    "bi_s1_leader": [
        ("left_arm_port", "/dev/ttyACM2"),
        ("right_arm_port", "/dev/ttyACM3"),
    ],
}


class DynamicParamsForm(QWidget):
    """动态子参数表单：set_schema 按类型重建行，values 导出 CLI 参数值。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._edits: dict[str, QLineEdit] = {}

    def set_schema(self, params: list[tuple[str, str]]) -> None:
        old_values = self.values()
        while self._form.rowCount():
            self._form.removeRow(0)
        self._edits = {}
        for name, default in params:
            edit = QLineEdit(old_values.get(name, default))
            # 与其他表单字段一致的尺寸策略，保证整列等宽对齐
            edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            self._edits[name] = edit
            self._form.addRow(name, edit)

    def values(self) -> dict[str, str]:
        return {name: edit.text().strip() for name, edit in self._edits.items() if edit.text().strip()}
