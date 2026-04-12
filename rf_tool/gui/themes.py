"""
Application themes for RF System Tool.

Each theme has:
  - ``qss``       : Qt Style Sheet applied to QApplication
  - ``canvas_bg`` : CSS hex color for the canvas background brush
"""
from __future__ import annotations

from typing import Dict

THEMES: Dict[str, Dict[str, str]] = {
    "Dark": {
        "qss": """
            QMainWindow, QWidget { background: #1A1A2E; color: #E0E0FF; }
            QMenuBar { background: #16213E; color: #E0E0FF; }
            QMenuBar::item:selected { background: #3A7BD5; }
            QMenu { background: #16213E; border: 1px solid #3A7BD5; color: #E0E0FF; }
            QMenu::item:selected { background: #3A7BD5; }
            QDockWidget { background: #1A1A2E; color: #E0E0FF; }
            QDockWidget::title { background: #16213E; padding: 4px; }
            QLabel { color: #E0E0FF; }
            QToolBar { background: #16213E; border: none; }
            QScrollBar { background: #1A1A2E; }
            QSplitter::handle { background: #3A7BD5; }
        """,
        "canvas_bg": "#1A1A2E",
    },
    "Light": {
        "qss": """
            QMainWindow, QWidget { background: #F0F0F0; color: #1A1A2E; }
            QMenuBar { background: #E0E0E0; color: #1A1A2E; }
            QMenuBar::item:selected { background: #A0C4FF; }
            QMenu { background: #FFFFFF; border: 1px solid #AAAAAA; color: #1A1A2E; }
            QMenu::item:selected { background: #A0C4FF; }
            QDockWidget { background: #F0F0F0; color: #1A1A2E; }
            QDockWidget::title { background: #DCDCDC; padding: 4px; }
            QLabel { color: #1A1A2E; }
            QToolBar { background: #E0E0E0; border: none; }
            QScrollBar { background: #F0F0F0; }
            QPushButton { background: #D0D8E8; color: #1A1A2E;
                          border: 1px solid #AAAAAA; padding: 4px 8px;
                          border-radius: 3px; }
            QPushButton:hover { background: #B0C4DE; }
            QTabWidget::pane { border: 1px solid #CCCCCC; }
            QTabBar::tab { background: #D0D0D0; color: #1A1A2E;
                           padding: 4px 12px; }
            QTabBar::tab:selected { background: #4080C0; color: white; }
        """,
        "canvas_bg": "#FFFFFF",
    },
    "Midnight Blue": {
        "qss": """
            QMainWindow, QWidget { background: #0D1B2A; color: #C8D8E8; }
            QMenuBar { background: #0A1520; color: #C8D8E8; }
            QMenuBar::item:selected { background: #1C4E80; }
            QMenu { background: #0A1520; border: 1px solid #1C4E80; color: #C8D8E8; }
            QMenu::item:selected { background: #1C4E80; }
            QDockWidget { background: #0D1B2A; color: #C8D8E8; }
            QDockWidget::title { background: #0A1520; padding: 4px; }
            QLabel { color: #C8D8E8; }
            QToolBar { background: #0A1520; border: none; }
            QPushButton { background: #1C3A5C; color: #C8D8E8;
                          border: 1px solid #2A5080; padding: 4px 8px;
                          border-radius: 3px; }
            QPushButton:hover { background: #2A5A8C; }
            QTabWidget::pane { border: 1px solid #1C4E80; }
            QTabBar::tab { background: #0D1B2A; color: #C8D8E8;
                           padding: 4px 12px; }
            QTabBar::tab:selected { background: #1C4E80; color: white; }
        """,
        "canvas_bg": "#0D1B2A",
    },
    "High Contrast": {
        "qss": """
            QMainWindow, QWidget { background: #000000; color: #FFFFFF; }
            QMenuBar { background: #000000; color: #FFFF00; }
            QMenuBar::item:selected { background: #FF0000; }
            QMenu { background: #000000; border: 1px solid #FFFF00; color: #FFFFFF; }
            QMenu::item:selected { background: #FF0000; }
            QDockWidget { background: #000000; color: #FFFFFF; }
            QDockWidget::title { background: #1A1A1A; padding: 4px; }
            QLabel { color: #FFFFFF; }
            QToolBar { background: #000000; border: none; }
            QPushButton { background: #1A1A1A; color: #FFFF00;
                          border: 2px solid #FFFF00; padding: 4px 8px;
                          border-radius: 3px; }
            QPushButton:hover { background: #333300; }
            QTabWidget::pane { border: 2px solid #FFFF00; }
            QTabBar::tab { background: #000000; color: #FFFF00;
                           padding: 4px 12px; border: 1px solid #888800; }
            QTabBar::tab:selected { background: #333300; color: #FFFFFF; }
        """,
        "canvas_bg": "#000000",
    },
}
