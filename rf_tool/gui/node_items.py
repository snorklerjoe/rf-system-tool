"""
Custom QGraphicsItem subclasses for each RF block type.

Each block type has a distinct schematic symbol painted using QPainter.
All block items share a common base (BlockItem) that handles:
  - Selection, hover, drag
  - Port rendering and hit-testing
  - Label display
  - Power-limit warning coloring
"""
from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsObject, QGraphicsTextItem,
    QStyleOptionGraphicsItem, QWidget,
)
from PySide6.QtGui import (
    QPainter, QPainterPath, QPen, QBrush, QColor, QFont,
    QPolygonF, QFontMetricsF, QLinearGradient,
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QObject

from rf_tool.models.rf_block import RFBlock, Port

if TYPE_CHECKING:
    pass


# ======================================================================= #
# Port graphics item                                                       #
# ======================================================================= #

PORT_RADIUS = 6.0
PORT_HIT_RADIUS = 10.0


class PortItem(QGraphicsObject):
    """Visual dot for a block port (input or output)."""

    # Signals
    connection_started = Signal(object)  # PortItem
    connection_finished = Signal(object)  # PortItem

    def __init__(self, port: Port, parent: "BlockItem"):
        super().__init__(parent)
        self.port = port
        self.setAcceptHoverEvents(True)
        self._hovered = False
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setCursor(Qt.CrossCursor)

    def boundingRect(self) -> QRectF:
        r = PORT_HIT_RADIUS
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget):
        color = QColor("#00DDFF") if self.port.direction == "output" else QColor("#FFAA00")
        if self._hovered:
            color = color.lighter(140)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.white, 1.0))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QRectF(-PORT_RADIUS, -PORT_RADIUS, 2 * PORT_RADIUS, 2 * PORT_RADIUS))

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.connection_started.emit(self)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.connection_finished.emit(self)
        super().mouseReleaseEvent(event)


# ======================================================================= #
# BlockItem base                                                           #
# ======================================================================= #

class BlockItem(QGraphicsObject):
    """
    Base class for all block graphics items.

    Subclasses should override:
        BLOCK_W, BLOCK_H  - bounding box size
        paint_shape()     - draw the schematic symbol
    """

    BLOCK_W = 80.0
    BLOCK_H = 60.0
    LABEL_FONT_SIZE = 9

    block_moved = Signal(str, float, float)   # block_id, x, y
    block_double_clicked = Signal(str)         # block_id
    block_clicked = Signal(str)                # block_id

    def __init__(self, block: RFBlock, parent=None):
        super().__init__(parent)
        self.block = block
        self._power_warning: str = "ok"   # "ok", "high", "low"

        # Qt item flags
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Position from block model
        self.setPos(block.x, block.y)

        # Label
        self._label_item = QGraphicsTextItem(block.label, self)
        self._label_item.setDefaultTextColor(Qt.white)
        font = QFont("Arial", self.LABEL_FONT_SIZE)
        self._label_item.setFont(font)
        self._position_label()

        # Port items
        self._port_items: List[PortItem] = []
        self._create_port_items()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _position_label(self) -> None:
        fm = QFontMetricsF(self._label_item.font())
        text_w = fm.horizontalAdvance(self.block.label)
        self._label_item.setPos(
            self.BLOCK_W / 2 - text_w / 2,
            self.BLOCK_H + 2,
        )

    def _create_port_items(self) -> None:
        """Create PortItem children at evenly-spaced positions."""
        # Input ports on left edge
        inputs = self.block.input_ports
        for i, port in enumerate(inputs):
            y = self.BLOCK_H * (i + 1) / (len(inputs) + 1)
            pi = PortItem(port, self)
            pi.setPos(0.0, y)
            self._port_items.append(pi)

        # Output ports on right edge
        outputs = self.block.output_ports
        for i, port in enumerate(outputs):
            y = self.BLOCK_H * (i + 1) / (len(outputs) + 1)
            pi = PortItem(port, self)
            pi.setPos(self.BLOCK_W, y)
            self._port_items.append(pi)

    def rebuild_ports(self) -> None:
        """Recreate port graphics after block port definitions change."""
        for pi in self._port_items:
            pi.setParentItem(None)
            if pi.scene():
                pi.scene().removeItem(pi)
        self._port_items = []
        self._create_port_items()

    def get_port_item(self, port_name: str) -> Optional[PortItem]:
        for pi in self._port_items:
            if pi.port.name == port_name:
                return pi
        return None

    def set_power_warning(self, status: str) -> None:
        """Set visual warning: 'ok', 'high', 'low'."""
        self._power_warning = status
        self.update()

    def update_label(self) -> None:
        self._label_item.setPlainText(self.block.label)
        self._position_label()

    # ------------------------------------------------------------------ #
    # QGraphicsItem overrides                                              #
    # ------------------------------------------------------------------ #
    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.BLOCK_W, self.BLOCK_H)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget):
        painter.setRenderHint(QPainter.Antialiasing)

        # Selection outline
        if self.isSelected():
            painter.setPen(QPen(QColor("#FFFF00"), 2.5))
        elif self._power_warning == "high":
            painter.setPen(QPen(QColor("#FF3333"), 3.0))
        elif self._power_warning == "low":
            painter.setPen(QPen(QColor("#FF8800"), 3.0))
        else:
            painter.setPen(QPen(QColor("#AAAACC"), 1.2))

        self.paint_shape(painter)

        if self.block.comment_mode != "active":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(90, 90, 90, 140)))
            painter.drawRoundedRect(self.boundingRect(), 6, 6)
            painter.setPen(QPen(QColor("#FF3333"), 3.0))
            if self.block.comment_mode == "out":
                painter.drawLine(QPointF(6, 6), QPointF(self.BLOCK_W - 6, self.BLOCK_H - 6))
                painter.drawLine(QPointF(self.BLOCK_W - 6, 6), QPointF(6, self.BLOCK_H - 6))
            elif self.block.comment_mode == "through":
                painter.drawLine(QPointF(6, self.BLOCK_H / 2), QPointF(self.BLOCK_W - 6, self.BLOCK_H / 2))

    def paint_shape(self, painter: QPainter) -> None:
        """Subclasses override this to draw their schematic symbol."""
        color = QColor(self.block.color)
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(self.boundingRect(), 6, 6)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.block.x = value.x()
            self.block.y = value.y()
            self.block_moved.emit(self.block.block_id, value.x(), value.y())
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        self.block_double_clicked.emit(self.block.block_id)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        self.block_clicked.emit(self.block.block_id)
        super().mousePressEvent(event)


# ======================================================================= #
# Amplifier / Gain block – triangle pointing right                        #
# ======================================================================= #

class AmplifierItem(BlockItem):
    BLOCK_W = 80.0
    BLOCK_H = 60.0

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        color = QColor(self.block.color)
        gradient = QLinearGradient(0, 0, w, 0)
        gradient.setColorAt(0, color.darker(130))
        gradient.setColorAt(1, color.lighter(120))
        painter.setBrush(QBrush(gradient))
        # Triangle pointing right
        tri = QPolygonF([QPointF(0, 0), QPointF(w, h / 2), QPointF(0, h)])
        painter.drawPolygon(tri)
        # Label "A" in center
        painter.setPen(QPen(Qt.white, 1.0))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(QRectF(5, 10, w * 0.6, h - 20), Qt.AlignCenter, "▶")


# ======================================================================= #
# Attenuator – resistor-pi symbol (rectangle with pi)                    #
# ======================================================================= #

class AttenuatorItem(BlockItem):
    BLOCK_W = 70.0
    BLOCK_H = 50.0

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        painter.setBrush(QBrush(QColor(self.block.color)))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        # Draw "pi" network symbol lines
        painter.setPen(QPen(Qt.white, 1.5))
        # Horizontal through-line
        painter.drawLine(QPointF(8, h / 2), QPointF(w - 8, h / 2))
        # Two vertical shunt lines
        painter.drawLine(QPointF(w * 0.3, h * 0.25), QPointF(w * 0.3, h * 0.75))
        painter.drawLine(QPointF(w * 0.7, h * 0.25), QPointF(w * 0.7, h * 0.75))
        painter.setFont(QFont("Arial", 7))
        painter.drawText(QRectF(0, h - 16, w, 14), Qt.AlignCenter,
                         f"{abs(self.block.gain_db):.0f} dB")


# ======================================================================= #
# Mixer – circle with X                                                   #
# ======================================================================= #

class MixerItem(BlockItem):
    BLOCK_W = 80.0
    BLOCK_H = 80.0

    def _create_port_items(self) -> None:
        """Mixer: RF on left-center, LO on bottom-center, IF on right-center."""
        from rf_tool.models.rf_block import Port

        self._port_items = []
        # RF input (left side)
        rf_ports = [p for p in self.block.input_ports if p.name == "RF"]
        lo_ports = [p for p in self.block.input_ports if p.name == "LO"]
        if_ports = [p for p in self.block.output_ports if p.name == "IF"]

        for port in rf_ports:
            pi = PortItem(port, self)
            pi.setPos(0.0, self.BLOCK_H / 2)
            self._port_items.append(pi)
        for port in lo_ports:
            pi = PortItem(port, self)
            pi.setPos(self.BLOCK_W / 2, self.BLOCK_H)
            self._port_items.append(pi)
        for port in if_ports:
            pi = PortItem(port, self)
            pi.setPos(self.BLOCK_W, self.BLOCK_H / 2)
            self._port_items.append(pi)

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        cx, cy = w / 2, h / 2
        r = min(w, h) * 0.44
        painter.setBrush(QBrush(QColor(self.block.color)))
        painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.setPen(QPen(Qt.white, 2.0))
        d = r * 0.55
        painter.drawLine(QPointF(cx - d, cy - d), QPointF(cx + d, cy + d))
        painter.drawLine(QPointF(cx + d, cy - d), QPointF(cx - d, cy + d))
        # Port labels
        painter.setFont(QFont("Arial", 7))
        painter.drawText(QRectF(2, cy - 8, 16, 14), Qt.AlignCenter, "RF")
        painter.drawText(QRectF(cx - 8, h - 16, 16, 14), Qt.AlignCenter, "LO")
        painter.drawText(QRectF(w - 20, cy - 8, 18, 14), Qt.AlignCenter, "IF")


# ======================================================================= #
# SparBlock – S-parameter box                                             #
# ======================================================================= #

class SparBlockItem(BlockItem):
    BLOCK_W = 80.0
    BLOCK_H = 60.0

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        painter.setBrush(QBrush(QColor(self.block.color)))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        painter.setPen(QPen(Qt.white, 1.0))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "[S]")


# ======================================================================= #
# TransferFnBlock                                                         #
# ======================================================================= #

class TransferFnItem(BlockItem):
    BLOCK_W = 90.0
    BLOCK_H = 55.0

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        painter.setBrush(QBrush(QColor(self.block.color)))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        painter.setPen(QPen(Qt.white, 1.0))
        # Draw H(s) text with fraction line
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(QRectF(0, 4, w, h / 2 - 2), Qt.AlignCenter, "N(s)")
        painter.drawLine(QPointF(w * 0.2, h / 2), QPointF(w * 0.8, h / 2))
        painter.drawText(QRectF(0, h / 2 + 2, w, h / 2 - 4), Qt.AlignCenter, "D(s)")


# ======================================================================= #
# LPF item                                                                #
# ======================================================================= #

class LPFItem(BlockItem):
    BLOCK_W = 80.0
    BLOCK_H = 55.0

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        painter.setBrush(QBrush(QColor(self.block.color)))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        painter.setPen(QPen(Qt.white, 1.5))
        # Draw LPF frequency response curve
        pts = []
        for i in range(50):
            x = w * 0.1 + w * 0.8 * i / 49
            xn = i / 49  # normalized 0..1
            # Simple sigmoid-like curve for LPF
            y_norm = 1.0 / (1 + (xn * 4) ** 4)
            y = h * 0.15 + h * 0.6 * (1 - y_norm)
            pts.append(QPointF(x, h - y))
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        painter.drawPath(path)
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(QRectF(0, 0, w, h * 0.3), Qt.AlignCenter, "LPF")


# ======================================================================= #
# HPF item                                                                #
# ======================================================================= #

class HPFItem(BlockItem):
    BLOCK_W = 80.0
    BLOCK_H = 55.0

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        painter.setBrush(QBrush(QColor(self.block.color)))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        painter.setPen(QPen(Qt.white, 1.5))
        pts = []
        for i in range(50):
            x = w * 0.1 + w * 0.8 * i / 49
            xn = i / 49
            y_norm = 1.0 / (1 + ((1 - xn) * 4) ** 4)
            y = h * 0.15 + h * 0.6 * (1 - y_norm)
            pts.append(QPointF(x, h - y))
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        painter.drawPath(path)
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(QRectF(0, 0, w, h * 0.3), Qt.AlignCenter, "HPF")


# ======================================================================= #
# PowerSplitter / Combiner                                                #
# ======================================================================= #

class SplitterItem(BlockItem):
    BLOCK_W = 70.0
    BLOCK_H = 70.0

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        painter.setBrush(QBrush(QColor(self.block.color)))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        painter.setPen(QPen(Qt.white, 1.5))
        n = self.block.n_ways
        cx = w / 2
        cy = h / 2
        if not self.block.is_combiner:
            # Draw lines from left port to N right ports
            painter.drawLine(QPointF(6, cy), QPointF(cx, cy))
            for i in range(n):
                y_out = h * (i + 1) / (n + 1)
                painter.drawLine(QPointF(cx, cy), QPointF(w - 6, y_out))
        else:
            # Reverse
            for i in range(n):
                y_in = h * (i + 1) / (n + 1)
                painter.drawLine(QPointF(6, y_in), QPointF(cx, cy))
            painter.drawLine(QPointF(cx, cy), QPointF(w - 6, cy))
        painter.setFont(QFont("Arial", 7))
        painter.drawText(QRectF(0, h - 14, w, 12), Qt.AlignCenter,
                         f"{n}-way {'C' if self.block.is_combiner else 'S'}")


# ======================================================================= #
# Switch                                                                  #
# ======================================================================= #

class SwitchItem(BlockItem):
    BLOCK_W = 80.0
    BLOCK_H = 60.0

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        color = QColor(self.block.color)
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        painter.setPen(QPen(Qt.white, 1.5))
        # Draw switch symbol
        if self.block.topology == "1x2":
            # Input on left, two outputs on right
            mid_y = h / 2
            y0 = h * 0.3
            y1 = h * 0.7
            painter.drawLine(QPointF(8, mid_y), QPointF(w * 0.35, mid_y))
            # Active path (bold)
            active_y = y0 if self.block.active_port == 0 else y1
            inactive_y = y1 if self.block.active_port == 0 else y0
            painter.setPen(QPen(Qt.white, 2.5))
            painter.drawLine(QPointF(w * 0.35, mid_y), QPointF(w * 0.65, active_y))
            painter.setPen(QPen(QColor(120, 120, 120), 1.0, Qt.DashLine))
            painter.drawLine(QPointF(w * 0.35, mid_y), QPointF(w * 0.65, inactive_y))
            painter.setPen(QPen(Qt.white, 1.5))
            painter.drawLine(QPointF(w * 0.65, y0), QPointF(w - 8, y0))
            painter.drawLine(QPointF(w * 0.65, y1), QPointF(w - 8, y1))
        else:
            # 2x1
            y0 = h * 0.3
            y1 = h * 0.7
            mid_y = h / 2
            painter.drawLine(QPointF(8, y0), QPointF(w * 0.35, y0))
            painter.drawLine(QPointF(8, y1), QPointF(w * 0.35, y1))
            active_y = y0 if self.block.active_port == 0 else y1
            inactive_y = y1 if self.block.active_port == 0 else y0
            painter.setPen(QPen(Qt.white, 2.5))
            painter.drawLine(QPointF(w * 0.35, active_y), QPointF(w * 0.65, mid_y))
            painter.setPen(QPen(QColor(120, 120, 120), 1.0, Qt.DashLine))
            painter.drawLine(QPointF(w * 0.35, inactive_y), QPointF(w * 0.65, mid_y))
            painter.setPen(QPen(Qt.white, 1.5))
            painter.drawLine(QPointF(w * 0.65, mid_y), QPointF(w - 8, mid_y))

    def mouseDoubleClickEvent(self, event):
        """Double-click toggles the switch and emits the standard signal."""
        self.block.toggle_state()
        self.update()
        self.block_double_clicked.emit(self.block.block_id)


# ======================================================================= #
# Source                                                                  #
# ======================================================================= #

class SourceItem(BlockItem):
    BLOCK_W = 65.0
    BLOCK_H = 65.0

    def _create_port_items(self) -> None:
        """Source has no input ports; output port on right."""
        self._port_items = []
        for port in self.block.output_ports:
            pi = PortItem(port, self)
            pi.setPos(self.BLOCK_W, self.BLOCK_H / 2)
            self._port_items.append(pi)

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        cx, cy = w / 2, h / 2
        r = min(w, h) * 0.42
        painter.setBrush(QBrush(QColor(self.block.color)))
        painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.setPen(QPen(Qt.white, 1.5))
        # Sine wave inside circle
        pts = []
        for i in range(40):
            t = i / 39
            x = cx - r * 0.7 + t * r * 1.4
            y = cy - math.sin(t * 4 * math.pi) * r * 0.35
            pts.append(QPointF(x, y))
        path = QPainterPath()
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        painter.drawPath(path)


# ======================================================================= #
# Sink                                                                    #
# ======================================================================= #

class SinkItem(BlockItem):
    BLOCK_W = 65.0
    BLOCK_H = 65.0

    def _create_port_items(self) -> None:
        """Sink has only one input port on the left."""
        self._port_items = []
        for port in self.block.input_ports:
            pi = PortItem(port, self)
            pi.setPos(0.0, self.BLOCK_H / 2)
            self._port_items.append(pi)

    def paint_shape(self, painter: QPainter) -> None:
        w, h = self.BLOCK_W, self.BLOCK_H
        painter.setBrush(QBrush(QColor(self.block.color)))
        painter.drawRoundedRect(QRectF(8, 18, w - 16, h - 36), 5, 5)
        painter.setPen(QPen(Qt.white, 1.5))
        lead_y = h / 2
        painter.drawLine(QPointF(0, lead_y), QPointF(8, lead_y))
        painter.drawLine(QPointF(w - 8, lead_y), QPointF(w, lead_y))
        zig = [
            QPointF(12, lead_y), QPointF(18, lead_y - 8), QPointF(24, lead_y + 8),
            QPointF(30, lead_y - 8), QPointF(36, lead_y + 8), QPointF(42, lead_y - 8),
            QPointF(48, lead_y + 8), QPointF(54, lead_y - 8),
        ]
        path = QPainterPath()
        path.moveTo(zig[0])
        for p in zig[1:]:
            path.lineTo(p)
        painter.drawPath(path)

    def mouseDoubleClickEvent(self, event):
        """Double-click opens the spectrum viewer."""
        self.block_double_clicked.emit(self.block.block_id)


# ======================================================================= #
# Annotation (free-text label)                                            #
# ======================================================================= #

class AnnotationItem(QGraphicsTextItem):
    """A movable, editable text annotation on the canvas."""

    def __init__(self, text: str = "Note", x: float = 0, y: float = 0,
                 font_name: str = "Arial", font_size: int = 10,
                 color: str = "#FFFFFF"):
        super().__init__(text)
        self.setPos(x, y)
        self.setDefaultTextColor(QColor(color))
        self.setFont(QFont(font_name, font_size))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setTextInteractionFlags(Qt.TextEditorInteraction)

    def to_dict(self) -> dict:
        return {
            "text": self.toPlainText(),
            "x": self.pos().x(),
            "y": self.pos().y(),
            "font": self.font().family(),
            "font_size": self.font().pointSize(),
            "color": self.defaultTextColor().name(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotationItem":
        return cls(
            text=d.get("text", ""),
            x=d.get("x", 0),
            y=d.get("y", 0),
            font_name=d.get("font", "Arial"),
            font_size=d.get("font_size", 10),
            color=d.get("color", "#FFFFFF"),
        )


# ======================================================================= #
# Factory: block model → BlockItem subclass                               #
# ======================================================================= #

_ITEM_MAP = {
    "Amplifier":      AmplifierItem,
    "Attenuator":     AttenuatorItem,
    "Mixer":          MixerItem,
    "SparBlock":      SparBlockItem,
    "TransferFnBlock": TransferFnItem,
    "LowPassFilter":  LPFItem,
    "HighPassFilter": HPFItem,
    "PowerSplitter":  SplitterItem,
    "PowerCombiner":  SplitterItem,
    "Switch":         SwitchItem,
    "Source":         SourceItem,
    "Sink":           SinkItem,
}


def create_block_item(block: RFBlock) -> BlockItem:
    """Return the correct BlockItem subclass for the given RFBlock."""
    cls = _ITEM_MAP.get(block.BLOCK_TYPE, BlockItem)
    return cls(block)
