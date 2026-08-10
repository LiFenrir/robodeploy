"""全局深色现代主题：Fusion 风格 + QSS。apply_theme 在 QApplication 创建后调用。"""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

_QSS = """
* { font-size: 13px; }
QMainWindow, QWidget { background: #181b20; color: #d7dce2; }
QGroupBox {
    background: #21252c; border: 1px solid #2e333c; border-radius: 8px;
    margin-top: 14px; padding: 10px 8px 8px 8px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #9aa4b0; }
QPushButton {
    background: #2b313a; border: 1px solid #3a414c; border-radius: 6px; padding: 6px 14px;
}
QPushButton:hover { background: #343b46; border-color: #4a5568; }
QPushButton:pressed { background: #2563eb; border-color: #2563eb; color: #fff; }
QPushButton:disabled { background: #23262c; color: #5c6470; border-color: #2b2f36; }
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget, QTableWidget {
    background: #121519; border: 1px solid #333a44; border-radius: 6px; padding: 5px 8px;
    selection-background-color: #2563eb;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #2563eb; }
QComboBox::drop-down {
    border: none; width: 24px;
    border-top-right-radius: 6px; border-bottom-right-radius: 6px; background: #343b46;
}
QComboBox::drop-down:hover { background: #3d4550; }
QComboBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid #c8d0da;
}
QComboBox QAbstractItemView { background: #21252c; border: 1px solid #333a44; }
/* 相机槽删除按钮：醒目红叉 */
QPushButton#slotRemove {
    background: transparent; border: 1px solid #4a3038; border-radius: 6px;
    color: #f87171; font-size: 15px; font-weight: 700; padding: 2px 0;
}
QPushButton#slotRemove:hover { background: #4a3038; color: #fca5a5; }
QTabWidget::pane { border: 1px solid #2e333c; border-radius: 8px; top: -1px; }
QTabBar::tab {
    background: transparent; padding: 8px 20px; margin: 4px 2px 0 2px;
    border-top-left-radius: 8px; border-top-right-radius: 8px; color: #9aa4b0;
}
QTabBar::tab:selected { background: #21252c; color: #fff; }
QTabBar::tab:hover:!selected { color: #d7dce2; }
QCheckBox { spacing: 6px; }
QSlider::groove:horizontal { height: 4px; background: #333a44; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; background: #2563eb;
}
QScrollArea { border: none; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #3a414c; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #4a5568; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #3a414c; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QHeaderView::section {
    background: #21252c; border: none; border-bottom: 1px solid #333a44; padding: 6px;
}
QTableWidget::item:selected { background: #2563eb; }
QSplitter::handle { background: #2e333c; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical { height: 2px; }
QToolTip { background: #21252c; color: #d7dce2; border: 1px solid #333a44; padding: 4px 8px; }
"""


def apply_theme(app: QApplication) -> None:
    """应用深色现代主题（Fusion + QSS + 调色板兜底）。"""
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#181b20"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#d7dce2"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#121519"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#d7dce2"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2b313a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#d7dce2"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#5c6470"))
    app.setPalette(palette)
    app.setStyleSheet(_QSS)
