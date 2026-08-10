"""关节状态曲线组件（pyqtgraph 可选，未安装时降级为占位提示）。"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
except ImportError:
    pg = None


class StatePlot(QWidget):
    """关节角实时曲线：append(names, values) 追加一组采样。"""

    def __init__(self, max_points: int = 600, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._max_points = max_points
        self._curves = []
        self._names: list[str] = []

        if pg is None:
            label = QLabel("pyqtgraph 未安装，曲线功能不可用\npip install pyqtgraph")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self._plot = None
            return

        self._plot = pg.PlotWidget()
        self._plot.addLegend()
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot)

    def set_joints(self, names: list[str]) -> None:
        if pg is None:
            return
        self._names = list(names)
        self._plot.clear()
        self._curves = [self._plot.plot(pen=pg.mkPen(i, width=2), name=n) for i, n in enumerate(self._names)]
        self._data = [[] for _ in self._names]

    def append(self, values: list[float]) -> None:
        if pg is None or not self._curves or len(values) != len(self._curves):
            return
        for curve, series, v in zip(self._curves, self._data, values, strict=True):
            series.append(v)
            del series[: max(0, len(series) - self._max_points)]
            curve.setData(series)
