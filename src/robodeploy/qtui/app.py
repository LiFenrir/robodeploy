"""QApplication + MainWindow：QTabWidget 四个功能区。"""

import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget

from .camera_tab import CameraTab
from .dataset_tab import DatasetTab
from .deploy_tab import DeployTab
from .record_launcher import RecordLauncherTab
from .theme import apply_theme


class MainWindow(QMainWindow):
    """record_tab 非空时为采集会话模式（record_dataset --front_end=qt），否则为独立启动器模式。"""

    def __init__(self, record_tab: QWidget | None = None):
        super().__init__()
        self.setWindowTitle("robodeploy Qt 控制台")
        self.resize(1600, 950)

        self.camera_tab = CameraTab(self)
        self.dataset_tab = DatasetTab(self)
        self.deploy_tab = DeployTab(self)
        self.record_launcher = RecordLauncherTab(self) if record_tab is None else None

        tabs = QTabWidget()
        tabs.addTab(self.camera_tab, "相机调试")
        tabs.addTab(record_tab if record_tab is not None else self.record_launcher, "数据采集")
        tabs.addTab(self.dataset_tab, "数据检查")
        tabs.addTab(self.deploy_tab, "部署")
        if record_tab is not None:
            tabs.setCurrentIndex(1)
        self.setCentralWidget(tabs)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 重写方法
        for worker in self.camera_tab._workers.values():
            worker.stop()
        self.camera_tab._workers.clear()
        self.dataset_tab.shutdown()
        self.deploy_tab.shutdown()
        if self.record_launcher is not None:
            self.record_launcher.shutdown()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
