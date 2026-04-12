"""
Symbol editor dialog for hierarchical sub-circuit blocks.

Allows the user to define a custom schematic symbol by drawing
filled polygons and placing text labels.  Each pin of the sub-circuit
is represented as a draggable handle.

The dialog returns a ``symbol`` dict compatible with HierSubcircuit:

    {
        "shapes": [
            {"type": "polygon", "points": [[x, y], ...], "color": "#4080FF", "filled": True},
            {"type": "text", "text": "...", "x": float, "y": float,
             "color": "#FFFFFF", "size": 10},
        ],
        "pins": {"pinname": {"x": float, "y": float}, ...}
    }
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Any, Tuple

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QColorDialog, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsPolygonItem,
    QDialogButtonBox, QWidget, QSpinBox, QGroupBox, QSizePolicy,
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPolygonF, QFont,
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal


# ─── Draggable pin handle ────────────────────────────────────────────────────

class _PinHandle(QGraphicsEllipseItem):
    """A small draggable circle representing a pin position."""

    def __init__(self, name: str, x: float, y: float, scene: "SymbolScene"):
        r = 8.0
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.pin_name = name
        self.setPos(x, y)
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setBrush(QBrush(QColor("#00DDFF")))
        self.setPen(QPen(Qt.white, 1.5))
        self.setToolTip(f"Pin: {name}")
        self.setZValue(10)
        # label
        self._lbl = QGraphicsTextItem(name, self)
        self._lbl.setDefaultTextColor(Qt.white)
        self._lbl.setFont(QFont("Arial", 7))
        self._lbl.setPos(r + 2, -8)


# ─── Symbol scene ─────────────────────────────────────────────────────────────

class SymbolScene(QGraphicsScene):
    """Custom scene for the symbol drawing canvas."""

    def __init__(self):
        super().__init__()
        self.setSceneRect(-200, -150, 400, 300)
        self.setBackgroundBrush(QBrush(QColor("#1A1A2E")))
        self._shapes: List[Dict] = []
        self._polygon_points: List[QPointF] = []
        self._drawing_polygon: bool = False
        self._current_color: str = "#4080FF"
        self._temp_poly: Optional[QGraphicsPolygonItem] = None

    def set_color(self, color: str) -> None:
        self._current_color = color

    def start_polygon(self) -> None:
        self._drawing_polygon = True
        self._polygon_points = []

    def finish_polygon(self) -> None:
        if len(self._polygon_points) >= 3:
            pts = [[p.x(), p.y()] for p in self._polygon_points]
            self._shapes.append({
                "type": "polygon",
                "points": pts,
                "color": self._current_color,
                "filled": True,
            })
        self._drawing_polygon = False
        self._polygon_points = []
        if self._temp_poly:
            self.removeItem(self._temp_poly)
            self._temp_poly = None
        self._redraw()

    def add_text(self, text: str, color: str, size: int) -> None:
        self._shapes.append({
            "type": "text",
            "text": text,
            "x": 0.0,
            "y": 0.0,
            "color": color,
            "size": size,
        })
        self._redraw()

    def mousePressEvent(self, event) -> None:
        if self._drawing_polygon and event.button() == Qt.LeftButton:
            self._polygon_points.append(event.scenePos())
            self._update_temp_poly()
        super().mousePressEvent(event)

    def _update_temp_poly(self) -> None:
        if self._temp_poly:
            self.removeItem(self._temp_poly)
        if len(self._polygon_points) < 2:
            return
        poly = QPolygonF(self._polygon_points)
        self._temp_poly = QGraphicsPolygonItem(poly)
        self._temp_poly.setPen(QPen(QColor(self._current_color), 1.5, Qt.DashLine))
        self._temp_poly.setBrush(QBrush(QColor(self._current_color + "55")))
        self.addItem(self._temp_poly)

    def _redraw(self) -> None:
        """Redraw all committed shapes."""
        # Remove old committed shape items (non-handle, non-temp)
        for item in list(self.items()):
            if isinstance(item, (QGraphicsPolygonItem, QGraphicsTextItem)):
                if item is not self._temp_poly:
                    # Keep pin labels (child items)
                    if item.parentItem() is None:
                        self.removeItem(item)
        for shape in self._shapes:
            if shape["type"] == "polygon":
                poly = QPolygonF([QPointF(p[0], p[1]) for p in shape["points"]])
                item = QGraphicsPolygonItem(poly)
                item.setPen(QPen(QColor(shape["color"]), 1.5))
                if shape.get("filled", True):
                    item.setBrush(QBrush(QColor(shape["color"] + "AA")))
                item.setZValue(1)
                self.addItem(item)
            elif shape["type"] == "text":
                item = QGraphicsTextItem(shape["text"])
                item.setPos(shape["x"], shape["y"])
                item.setDefaultTextColor(QColor(shape["color"]))
                item.setFont(QFont("Arial", shape["size"]))
                item.setZValue(2)
                self.addItem(item)

    def get_symbol(self, pin_handles: List[_PinHandle]) -> Dict:
        pins = {h.pin_name: {"x": h.pos().x(), "y": h.pos().y()} for h in pin_handles}
        return {"shapes": list(self._shapes), "pins": pins}

    def load_symbol(self, symbol: Dict) -> None:
        self._shapes = list(symbol.get("shapes", []))
        self._redraw()


# ======================================================================= #
# SymbolEditorDialog                                                        #
# ======================================================================= #

class SymbolEditorDialog(QDialog):
    """
    Dialog for editing the visual symbol of a HierSubcircuit block.

    Parameters
    ----------
    pins : list of str
        Pin names to display as draggable handles.
    initial_symbol : dict, optional
        Pre-existing symbol definition to load.
    parent : QWidget, optional
    """

    def __init__(
        self,
        pins: Optional[List[str]] = None,
        initial_symbol: Optional[Dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Symbol Editor")
        self.resize(700, 500)
        self._pins: List[str] = pins or []
        self._pin_handles: List[_PinHandle] = []
        self._current_color: str = "#4080FF"
        self._symbol: Dict = {}

        self._build_ui()
        self._scene.load_symbol(initial_symbol or {})
        self._place_pin_handles(initial_symbol)

    # ------------------------------------------------------------------ #
    # Build UI                                                             #
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Canvas row ───────────────────────────────────────────────────
        canvas_row = QHBoxLayout()

        self._scene = SymbolScene()
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.Antialiasing)
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_row.addWidget(self._view, 3)

        # ── Control panel ────────────────────────────────────────────────
        ctrl = QWidget()
        ctrl_lay = QVBoxLayout(ctrl)
        ctrl_lay.setAlignment(Qt.AlignTop)
        ctrl.setMaximumWidth(200)

        # Color
        clr_grp = QGroupBox("Color")
        clr_lay = QHBoxLayout(clr_grp)
        self._color_btn = QPushButton("  ")
        self._color_btn.setStyleSheet(f"background: {self._current_color};")
        self._color_btn.clicked.connect(self._pick_color)
        clr_lay.addWidget(QLabel("Fill:"))
        clr_lay.addWidget(self._color_btn)
        ctrl_lay.addWidget(clr_grp)

        # Polygon
        poly_grp = QGroupBox("Polygon")
        poly_lay = QVBoxLayout(poly_grp)
        self._start_poly_btn = QPushButton("Add Polygon")
        self._start_poly_btn.clicked.connect(self._start_polygon)
        self._finish_poly_btn = QPushButton("Finish Polygon")
        self._finish_poly_btn.setEnabled(False)
        self._finish_poly_btn.clicked.connect(self._finish_polygon)
        poly_lay.addWidget(self._start_poly_btn)
        poly_lay.addWidget(self._finish_poly_btn)
        ctrl_lay.addWidget(poly_grp)

        # Text
        txt_grp = QGroupBox("Text")
        txt_lay = QVBoxLayout(txt_grp)
        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText("Label text…")
        self._text_size = QSpinBox()
        self._text_size.setRange(6, 48)
        self._text_size.setValue(10)
        add_txt_btn = QPushButton("Add Text")
        add_txt_btn.clicked.connect(self._add_text)
        txt_lay.addWidget(QLabel("Text:"))
        txt_lay.addWidget(self._text_input)
        txt_lay.addWidget(QLabel("Size:"))
        txt_lay.addWidget(self._text_size)
        txt_lay.addWidget(add_txt_btn)
        ctrl_lay.addWidget(txt_grp)

        canvas_row.addWidget(ctrl)
        layout.addLayout(canvas_row, 1)

        # ── Buttons ──────────────────────────────────────────────────────
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self._on_accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _place_pin_handles(self, symbol: Optional[Dict]) -> None:
        existing_pins = (symbol or {}).get("pins", {})
        spacing = 60.0
        start_y = -(len(self._pins) - 1) * spacing / 2
        for i, name in enumerate(self._pins):
            pos = existing_pins.get(name, {})
            x = pos.get("x", -100.0 if i % 2 == 0 else 100.0)
            y = pos.get("y", start_y + i * spacing)
            handle = _PinHandle(name, x, y, self._scene)
            self._scene.addItem(handle)
            self._pin_handles.append(handle)

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._current_color), self, "Pick Color")
        if color.isValid():
            self._current_color = color.name()
            self._color_btn.setStyleSheet(f"background: {self._current_color};")
            self._scene.set_color(self._current_color)

    def _start_polygon(self) -> None:
        self._scene.start_polygon()
        self._start_poly_btn.setEnabled(False)
        self._finish_poly_btn.setEnabled(True)

    def _finish_polygon(self) -> None:
        self._scene.finish_polygon()
        self._start_poly_btn.setEnabled(True)
        self._finish_poly_btn.setEnabled(False)

    def _add_text(self) -> None:
        text = self._text_input.text().strip()
        if text:
            self._scene.add_text(text, self._current_color, self._text_size.value())
            self._text_input.clear()

    def _on_accept(self) -> None:
        self._symbol = self._scene.get_symbol(self._pin_handles)
        self.accept()

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #
    def get_symbol(self) -> Dict:
        """Return the symbol dict after the dialog has been accepted."""
        return self._symbol
