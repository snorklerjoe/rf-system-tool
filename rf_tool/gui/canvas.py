"""
RF Canvas – QGraphicsScene + QGraphicsView for the node-based editor.

Handles:
  - Adding, removing, and moving block items
  - Drawing and tracking wire connections
  - Annotation placement
  - Routing signals through the graph
  - P2P start/end node selection
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsLineItem,
    QGraphicsPathItem, QMenu, QApplication,
)
from PySide6.QtGui import QPen, QColor, QPainterPath, QBrush, QTransform
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QObject

from rf_tool.models.rf_block import RFBlock
from rf_tool.models.signal import Signal as RFSignal
from rf_tool.gui.node_items import (
    BlockItem, PortItem, AnnotationItem, create_block_item,
)


# ======================================================================= #
# Wire item                                                                #
# ======================================================================= #

class WireItem(QGraphicsPathItem):
    """Curved wire connecting two PortItems."""

    def __init__(self,
                 src_port: PortItem,
                 dst_port: PortItem,
                 parent=None):
        super().__init__(parent)
        self.src_port = src_port
        self.dst_port = dst_port
        self.setPen(QPen(QColor("#00CCFF"), 2.0, Qt.SolidLine,
                         Qt.RoundCap, Qt.RoundJoin))
        self.setZValue(-1)
        self.update_path()

    def update_path(self) -> None:
        """Redraw the bezier wire between the two port positions."""
        p1 = self.src_port.scenePos()
        p2 = self.dst_port.scenePos()
        dx = abs(p2.x() - p1.x()) * 0.5

        path = QPainterPath()
        path.moveTo(p1)
        path.cubicTo(
            QPointF(p1.x() + dx, p1.y()),
            QPointF(p2.x() - dx, p2.y()),
            p2,
        )
        self.setPath(path)


class TempWireItem(QGraphicsPathItem):
    """Temporary wire drawn while the user is dragging a connection."""

    def __init__(self, start: QPointF, parent=None):
        super().__init__(parent)
        self.start = start
        self.end = start
        self.setPen(QPen(QColor("#88FFFF"), 2.0, Qt.DashLine))
        self.setZValue(10)

    def update_end(self, end: QPointF) -> None:
        self.end = end
        dx = abs(self.end.x() - self.start.x()) * 0.5
        path = QPainterPath()
        path.moveTo(self.start)
        path.cubicTo(
            QPointF(self.start.x() + dx, self.start.y()),
            QPointF(self.end.x() - dx, self.end.y()),
            self.end,
        )
        self.setPath(path)


# ======================================================================= #
# Canvas Scene                                                             #
# ======================================================================= #

class RFScene(QGraphicsScene):
    """
    QGraphicsScene that manages blocks, wires, and annotations.
    """

    connection_made = Signal(str, str, str, str)    # src_bid, src_port, dst_bid, dst_port
    connection_removed = Signal(str, str, str, str)
    block_selected = Signal(str)                     # block_id
    scene_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.setBackgroundBrush(QBrush(QColor("#1A1A2E")))

        # Data
        self._block_items: Dict[str, BlockItem] = {}   # block_id -> BlockItem
        self._wires: List[WireItem] = []
        self._connections: List[Dict] = []             # serialisable list

        # Wire drag state
        self._dragging_wire: Optional[TempWireItem] = None
        self._drag_src_port: Optional[PortItem] = None

        # P2P selection
        self._p2p_start: Optional[str] = None
        self._p2p_end: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Block management                                                     #
    # ------------------------------------------------------------------ #
    def add_block(self, block: RFBlock) -> BlockItem:
        item = create_block_item(block)
        self.addItem(item)
        self._block_items[block.block_id] = item

        # Connect port signals for wire dragging
        for port_item in item._port_items:
            port_item.connection_started.connect(self._on_port_drag_start)
            port_item.connection_finished.connect(self._on_port_drag_end)

        item.block_clicked.connect(self.block_selected.emit)
        self.scene_changed.emit()
        return item

    def remove_block(self, block_id: str) -> None:
        item = self._block_items.pop(block_id, None)
        if item is None:
            return
        # Remove connected wires
        wires_to_remove = [
            w for w in self._wires
            if (w.src_port.parentItem() is item or w.dst_port.parentItem() is item)
        ]
        for w in wires_to_remove:
            self._remove_wire(w)
        self.removeItem(item)
        self.scene_changed.emit()

    def get_block_item(self, block_id: str) -> Optional[BlockItem]:
        return self._block_items.get(block_id)

    def get_all_blocks(self) -> List[RFBlock]:
        return [item.block for item in self._block_items.values()]

    # ------------------------------------------------------------------ #
    # Wire management                                                      #
    # ------------------------------------------------------------------ #
    def add_wire(self, src_item: BlockItem, src_port_name: str,
                 dst_item: BlockItem, dst_port_name: str) -> Optional[WireItem]:
        src_port = src_item.get_port_item(src_port_name)
        dst_port = dst_item.get_port_item(dst_port_name)
        if src_port is None or dst_port is None:
            return None
        wire = WireItem(src_port, dst_port)
        self.addItem(wire)
        self._wires.append(wire)
        self._connections.append({
            "src_block_id": src_item.block.block_id,
            "src_port": src_port_name,
            "dst_block_id": dst_item.block.block_id,
            "dst_port": dst_port_name,
        })
        self.connection_made.emit(
            src_item.block.block_id, src_port_name,
            dst_item.block.block_id, dst_port_name,
        )
        self.scene_changed.emit()
        return wire

    def _remove_wire(self, wire: WireItem) -> None:
        if wire in self._wires:
            self._wires.remove(wire)
        # Remove from connections list
        src_bid = wire.src_port.parentItem().block.block_id
        dst_bid = wire.dst_port.parentItem().block.block_id
        self._connections = [
            c for c in self._connections
            if not (c["src_block_id"] == src_bid and c["dst_block_id"] == dst_bid
                    and c["src_port"] == wire.src_port.port.name
                    and c["dst_port"] == wire.dst_port.port.name)
        ]
        self.removeItem(wire)
        self.scene_changed.emit()

    # ------------------------------------------------------------------ #
    # Wire drag handlers                                                   #
    # ------------------------------------------------------------------ #
    def _on_port_drag_start(self, port_item: PortItem) -> None:
        if self._dragging_wire:
            return
        self._drag_src_port = port_item
        pos = port_item.scenePos()
        self._dragging_wire = TempWireItem(pos)
        self.addItem(self._dragging_wire)

    def _on_port_drag_end(self, port_item: PortItem) -> None:
        if self._dragging_wire is None or self._drag_src_port is None:
            return
        self.removeItem(self._dragging_wire)
        self._dragging_wire = None

        src = self._drag_src_port
        dst = port_item
        self._drag_src_port = None

        if src is dst:
            return
        if src.port.direction == dst.port.direction:
            return  # same direction: skip

        # Ensure output -> input order
        if src.port.direction == "input":
            src, dst = dst, src

        src_item = src.parentItem()
        dst_item = dst.parentItem()
        if src_item is dst_item:
            return

        self.add_wire(src_item, src.port.name, dst_item, dst.port.name)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_wire:
            self._dragging_wire.update_end(event.scenePos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging_wire and event.button() == Qt.LeftButton:
            # Check if we released over a port
            items = self.items(event.scenePos())
            dst_port = None
            for item in items:
                if isinstance(item, PortItem) and item is not self._drag_src_port:
                    dst_port = item
                    break
            if dst_port:
                self._on_port_drag_end(dst_port)
            else:
                # Cancel
                if self._dragging_wire:
                    self.removeItem(self._dragging_wire)
                    self._dragging_wire = None
                    self._drag_src_port = None
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------ #
    # Annotation management                                                #
    # ------------------------------------------------------------------ #
    def add_annotation(self, text: str = "Note", x: float = 0, y: float = 0,
                        font: str = "Arial", font_size: int = 10,
                        color: str = "#FFFFFF") -> AnnotationItem:
        item = AnnotationItem(text, x, y, font, font_size, color)
        self.addItem(item)
        self.scene_changed.emit()
        return item

    # ------------------------------------------------------------------ #
    # Update wires after block move                                        #
    # ------------------------------------------------------------------ #
    def update_all_wires(self) -> None:
        for wire in self._wires:
            wire.update_path()

    def mouseMoveEvent(self, event) -> None:  # noqa: F811
        self.update_all_wires()
        if self._dragging_wire:
            self._dragging_wire.update_end(event.scenePos())
        super().mouseMoveEvent(event)

    # ------------------------------------------------------------------ #
    # Signal propagation                                                   #
    # ------------------------------------------------------------------ #
    def propagate_signals(self) -> Dict[str, "RFSignal"]:
        """
        Propagate signals from Source blocks through the graph.

        Returns
        -------
        dict mapping block_id -> last Signal received at that block.
        """
        from rf_tool.blocks.components import Source
        signals_at: Dict[str, Dict[str, RFSignal]] = {}

        # Build adjacency: src_bid/src_port -> (dst_bid, dst_port)
        adj: Dict[Tuple, List[Tuple]] = {}
        for c in self._connections:
            key = (c["src_block_id"], c["src_port"])
            adj.setdefault(key, []).append((c["dst_block_id"], c["dst_port"]))

        # Find sources and seed
        queue: List[Tuple] = []
        for bid, item in self._block_items.items():
            if isinstance(item.block, Source):
                sig = item.block.generate()
                signals_at.setdefault(bid, {})[item.block.output_ports[0].name] = sig
                queue.append((bid, item.block.output_ports[0].name, sig))

        # BFS propagation
        visited = set()
        while queue:
            src_bid, src_port, sig = queue.pop(0)
            if (src_bid, src_port) in visited:
                continue
            visited.add((src_bid, src_port))
            for dst_bid, dst_port in adj.get((src_bid, src_port), []):
                dst_item = self._block_items.get(dst_bid)
                if dst_item is None:
                    continue
                result = dst_item.block.process(sig, dst_port)
                signals_at.setdefault(dst_bid, {})[dst_port] = sig

                # Power warning
                status = dst_item.block.check_power(sig.power_dbm)
                dst_item.set_power_warning(status)

                for out_port, out_sig in result.items():
                    signals_at.setdefault(dst_bid, {})[out_port] = out_sig
                    queue.append((dst_bid, out_port, out_sig))

        return signals_at

    # ------------------------------------------------------------------ #
    # P2P path finding                                                     #
    # ------------------------------------------------------------------ #
    def find_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """
        BFS from start_id to end_id following connections.

        Returns
        -------
        List of block_ids from start to end, or None if no path.
        """
        # Build adjacency on block level
        adj: Dict[str, List[str]] = {}
        for c in self._connections:
            adj.setdefault(c["src_block_id"], []).append(c["dst_block_id"])

        from collections import deque
        q: deque = deque([[start_id]])
        seen = {start_id}
        while q:
            path = q.popleft()
            node = path[-1]
            if node == end_id:
                return path
            for nxt in adj.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(path + [nxt])
        return None

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #
    def get_annotations(self) -> List[AnnotationItem]:
        return [item for item in self.items() if isinstance(item, AnnotationItem)]

    def get_connections(self) -> List[Dict]:
        return list(self._connections)

    def clear_scene(self) -> None:
        """Remove all items and reset state."""
        super().clear()
        self._block_items.clear()
        self._wires.clear()
        self._connections.clear()


# ======================================================================= #
# Canvas View                                                              #
# ======================================================================= #

class RFCanvasView(QGraphicsView):
    """QGraphicsView with zoom/pan support for the RF canvas."""

    def __init__(self, scene: RFScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 1.0

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._zoom *= factor
        self._zoom = max(0.1, min(self._zoom, 10.0))
        self.setTransform(QTransform().scale(self._zoom, self._zoom))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Delete:
            for item in self.scene().selectedItems():
                if isinstance(item, BlockItem):
                    self.scene().remove_block(item.block.block_id)
                elif isinstance(item, AnnotationItem):
                    self.scene().removeItem(item)
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom = min(self._zoom * 1.2, 10.0)
            self.setTransform(QTransform().scale(self._zoom, self._zoom))
        elif event.key() == Qt.Key_Minus:
            self._zoom = max(self._zoom / 1.2, 0.1)
            self.setTransform(QTransform().scale(self._zoom, self._zoom))
        elif event.key() == Qt.Key_0:
            self._zoom = 1.0
            self.setTransform(QTransform().scale(1.0, 1.0))
        else:
            super().keyPressEvent(event)
