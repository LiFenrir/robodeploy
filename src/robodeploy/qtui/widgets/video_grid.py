"""视频组件：CameraView 等比缩放显示 RGB 帧，VideoGrid 自适应网格布局。"""

import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


class CameraView(QFrame):
    """单路视频视图：QLabel 显示 QImage，自适应缩放（keepAspectRatio）。

    set_frame 输入 RGB uint8 (H, W, 3)。
    """

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("CameraView { background: #1e1e1e; border-radius: 6px; }")

        self._frame: np.ndarray | None = None

        self._image_label = QLabel("无信号")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("color: #888; background: transparent;")

        self._name_label = QLabel(name, self)
        self._name_label.setStyleSheet(
            "color: #fff; background: rgba(0, 0, 0, 128); padding: 2px 8px; border-radius: 4px;"
        )
        self._name_label.move(8, 8)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._image_label)

    def set_name(self, name: str) -> None:
        self._name_label.setText(name)

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._render()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 重写方法
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._frame is None:
            return
        h, w, c = self._frame.shape
        image = QImage(self._frame.data, w, h, c * w, QImage.Format.Format_RGB888)
        # copy 脱离 numpy 缓冲，防止数据被复用后画面撕裂
        pixmap = QPixmap.fromImage(image.copy())
        self._image_label.setPixmap(
            pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class VideoGrid(QWidget):
    """多路视频网格：列数按视图数量自适应；single_row=True 时全部排在一行。"""

    def __init__(self, parent: QWidget | None = None, single_row: bool = False):
        super().__init__(parent)
        self._views: dict[str, CameraView] = {}
        self._single_row = single_row
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(4, 4, 4, 4)

    def add_view(self, name: str) -> CameraView:
        if name in self._views:
            return self._views[name]
        view = CameraView(name, self)
        self._views[name] = view
        self._relayout()
        return view

    def remove_view(self, name: str) -> None:
        view = self._views.pop(name, None)
        if view is not None:
            self._grid.removeWidget(view)
            view.deleteLater()
            self._relayout()

    def set_frame(self, name: str, frame: np.ndarray) -> None:
        view = self._views.get(name)
        if view is not None:
            view.set_frame(frame)

    def view_names(self) -> list[str]:
        return list(self._views)

    def _relayout(self) -> None:
        for i in reversed(range(self._grid.count())):
            self._grid.takeAt(i)
        count = len(self._views)
        cols = count if self._single_row else max(1, math.ceil(math.sqrt(count)))
        for idx, view in enumerate(self._views.values()):
            self._grid.addWidget(view, idx // cols, idx % cols)
