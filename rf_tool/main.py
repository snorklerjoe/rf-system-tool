"""
RF System Tool – main entry point.
"""
from __future__ import annotations

import sys
import os


def main() -> None:
    # Set pyqtgraph to use PySide6 before any imports
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from rf_tool.gui.main_window import MainWindow

    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("RF System Tool")
    app.setOrganizationName("rf-system-tool")
    app.setStyle("Fusion")

    # Dark palette
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(30, 30, 46))
    palette.setColor(QPalette.WindowText,      QColor(220, 220, 255))
    palette.setColor(QPalette.Base,            QColor(20, 20, 35))
    palette.setColor(QPalette.AlternateBase,   QColor(35, 35, 55))
    palette.setColor(QPalette.ToolTipBase,     QColor(50, 50, 80))
    palette.setColor(QPalette.ToolTipText,     QColor(220, 220, 255))
    palette.setColor(QPalette.Text,            QColor(220, 220, 255))
    palette.setColor(QPalette.Button,          QColor(45, 45, 70))
    palette.setColor(QPalette.ButtonText,      QColor(220, 220, 255))
    palette.setColor(QPalette.BrightText,      QColor(255, 80, 80))
    palette.setColor(QPalette.Link,            QColor(80, 160, 255))
    palette.setColor(QPalette.Highlight,       QColor(50, 100, 200))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
