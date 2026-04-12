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

import math
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView,
    QGraphicsPathItem, QGraphicsSimpleTextItem, QGraphicsItem,
)
from PySide6.QtGui import QPen, QColor, QPainterPath, QBrush, QTransform
from PySide6.QtCore import Qt, QPointF, QPoint, Signal

from rf_tool.models.rf_block import RFBlock
from rf_tool.models.signal import Signal as RFSignal
from rf_tool.gui.node_items import (
    BlockItem, PortItem, AnnotationItem, create_block_item,
)

POWER_EPSILON_DBM = 1e-9
FREQUENCY_EPSILON_HZ = 1e-3
MIN_POWER_MW = 1e-300
MIN_PROPAGATION_ITERATIONS = 1000
# Multiplier chosen to allow multi-branch convergence while preventing runaway loops.
ITERATIONS_PER_CONNECTION = 40


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
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self._label = QGraphicsSimpleTextItem("-∞ dBm", self)
        self._label.setBrush(QColor("#BBBBFF"))
        self._label.setZValue(1)
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
        mid = path.pointAtPercent(0.5)
        self._label.setPos(mid.x() + 6, mid.y() - 14)

    def set_power_label(self, text: str, color: QColor) -> None:
        self._label.setText(text)
        self._label.setBrush(color)
        font = self._label.font()
        font.setBold(color == QColor("#FF3333"))
        self._label.setFont(font)

    def mousePressEvent(self, event) -> None:
        """Handle mouse click on wire."""
        super().mousePressEvent(event)
        # The scene will handle the selection state
        
    def itemChange(self, change, value):
        """Update visual state when selection changes."""
        if change == QGraphicsItem.ItemSelectedChange:
            if value:
                # Wire is now selected - make it more visible
                self.setPen(QPen(QColor("#FFFF00"), 3.5, Qt.SolidLine,
                               Qt.RoundCap, Qt.RoundJoin))
            else:
                # Wire is now deselected - return to normal
                self.setPen(QPen(QColor("#00CCFF"), 2.0, Qt.SolidLine,
                               Qt.RoundCap, Qt.RoundJoin))
        return super().itemChange(change, value)


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
    block_double_clicked = Signal(str)
    wire_selected = Signal(str, str, str, str)       # src_bid, src_port, dst_bid, dst_port
    blocks_selected = Signal(list)                   # list of block_ids (multi-select)
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
        
        # Connect selection changes
        self.selectionChanged.connect(self._on_selection_changed)

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
        item.block_double_clicked.connect(self.block_double_clicked.emit)
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

    def get_selected_wires(self) -> List[WireItem]:
        return [item for item in self.selectedItems() if isinstance(item, WireItem)]

    def select_all(self) -> None:
        for item in self.items():
            if isinstance(item, (BlockItem, AnnotationItem, WireItem)):
                item.setSelected(True)

    def _on_selection_changed(self) -> None:
        """Handle selection changes - emit wire_selected or blocks_selected signals."""
        selected_wires = [item for item in self.selectedItems() if isinstance(item, WireItem)]
        selected_blocks = [item for item in self.selectedItems() if isinstance(item, BlockItem)]

        if len(selected_wires) == 1 and not selected_blocks:
            wire = selected_wires[0]
            src_bid = wire.src_port.parentItem().block.block_id
            dst_bid = wire.dst_port.parentItem().block.block_id
            self.wire_selected.emit(src_bid, wire.src_port.port.name, dst_bid, wire.dst_port.port.name)
        elif len(selected_blocks) >= 2:
            self.blocks_selected.emit([item.block.block_id for item in selected_blocks])

    def rebuild_block_ports(self, block_id: str) -> None:
        item = self._block_items.get(block_id)
        if item is None:
            return
        connected = [c for c in self._connections if c["src_block_id"] == block_id or c["dst_block_id"] == block_id]
        for wire in list(self._wires):
            src_item = wire.src_port.parentItem()
            dst_item = wire.dst_port.parentItem()
            if src_item is item or dst_item is item:
                self._remove_wire(wire)
        item.rebuild_ports()
        for port_item in item._port_items:
            port_item.connection_started.connect(self._on_port_drag_start)
            port_item.connection_finished.connect(self._on_port_drag_end)
        for c in connected:
            src = self._block_items.get(c["src_block_id"])
            dst = self._block_items.get(c["dst_block_id"])
            if src and dst:
                self.add_wire(src, c["src_port"], dst, c["dst_port"])
        self.scene_changed.emit()

    # ------------------------------------------------------------------ #
    # Wire management                                                      #
    # ------------------------------------------------------------------ #
    def add_wire(self, src_item: BlockItem, src_port_name: str,
                 dst_item: BlockItem, dst_port_name: str) -> Optional[WireItem]:
        if any(
            c["src_block_id"] == src_item.block.block_id and
            c["src_port"] == src_port_name and
            c["dst_block_id"] == dst_item.block.block_id and
            c["dst_port"] == dst_port_name
            for c in self._connections
        ):
            return None
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
    @staticmethod
    def _signals_equivalent(a: Optional[RFSignal], b: Optional[RFSignal]) -> bool:
        if a is None or b is None:
            return a is b
        if abs(a.power_dbm - b.power_dbm) > POWER_EPSILON_DBM:
            return False
        if abs(a.carrier_frequency - b.carrier_frequency) > FREQUENCY_EPSILON_HZ:
            return False
        if len(a.spurs) != len(b.spurs):
            return False
        a_spurs = sorted(a.spurs, key=lambda s: (s.frequency, s.power_dbm))
        b_spurs = sorted(b.spurs, key=lambda s: (s.frequency, s.power_dbm))
        for a_spur, b_spur in zip(a_spurs, b_spurs):
            if abs(a_spur.frequency - b_spur.frequency) > FREQUENCY_EPSILON_HZ:
                return False
            if abs(a_spur.power_dbm - b_spur.power_dbm) > POWER_EPSILON_DBM:
                return False
        a_nf = a.get_noise_floor_dbm()
        b_nf = b.get_noise_floor_dbm()
        if not (a_nf is None and b_nf is None):
            if (a_nf is None) != (b_nf is None):
                return False
            if abs(a_nf - b_nf) > POWER_EPSILON_DBM:
                return False
        return True

    @staticmethod
    def _merge_signals(existing: Optional[RFSignal], incoming: RFSignal) -> RFSignal:
        if existing is None:
            return incoming.copy()
        tone_bins: List[Tuple[float, float]] = []

        def _add_tone(freq_hz: float, power_dbm: float) -> None:
            power_mw = 10.0 ** (power_dbm / 10.0)
            for i, (f_hz, p_mw) in enumerate(tone_bins):
                if abs(f_hz - freq_hz) < 1e-3:
                    tone_bins[i] = (f_hz, p_mw + power_mw)
                    return
            tone_bins.append((freq_hz, power_mw))

        _add_tone(existing.carrier_frequency, existing.power_dbm)
        for spur in existing.spurs:
            _add_tone(spur.frequency, spur.power_dbm)
        _add_tone(incoming.carrier_frequency, incoming.power_dbm)
        for spur in incoming.spurs:
            _add_tone(spur.frequency, spur.power_dbm)

        combined_tones = sorted(
            [(f_hz, 10.0 * math.log10(max(p_mw, MIN_POWER_MW))) for f_hz, p_mw in tone_bins],
            key=lambda x: x[1],
            reverse=True,
        )
        carrier_f, carrier_p = combined_tones[0]
        out = RFSignal(carrier_frequency=carrier_f, power_dbm=carrier_p, spurs=[])
        for f_hz, p_dbm in combined_tones[1:]:
            out.add_spur(f_hz, p_dbm)
        existing_nf = existing.get_noise_floor_dbm()
        incoming_nf = incoming.get_noise_floor_dbm()
        if existing_nf is None and incoming_nf is None:
            if existing.snr_db is None:
                out.snr_db = incoming.snr_db
            elif incoming.snr_db is None:
                out.snr_db = existing.snr_db
            else:
                existing_mw = 10.0 ** (existing.power_dbm / 10.0)
                incoming_mw = 10.0 ** (incoming.power_dbm / 10.0)
                noise_existing = existing_mw / (10.0 ** (existing.snr_db / 10.0))
                noise_incoming = incoming_mw / (10.0 ** (incoming.snr_db / 10.0))
                total_noise = noise_existing + noise_incoming
                total_signal = existing_mw + incoming_mw
                out.snr_db = 10.0 * math.log10(total_signal / max(total_noise, MIN_POWER_MW))
            return out

        noise_terms = []
        if existing_nf is not None:
            noise_terms.append(10.0 ** (existing_nf / 10.0))
        if incoming_nf is not None:
            noise_terms.append(10.0 ** (incoming_nf / 10.0))
        total_noise_mw = sum(noise_terms)
        out_noise_floor = 10.0 * math.log10(max(total_noise_mw, MIN_POWER_MW))
        out.set_noise_floor_dbm(out_noise_floor)
        return out

    @staticmethod
    def _wire_key(connection: Dict) -> Tuple[str, str, str, str]:
        return (
            connection["src_block_id"],
            connection["src_port"],
            connection["dst_block_id"],
            connection["dst_port"],
        )

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

        wire_power: Dict[Tuple[str, str, str, str], float] = {}
        for w in self._wires:
            key = (
                w.src_port.parentItem().block.block_id,
                w.src_port.port.name,
                w.dst_port.parentItem().block.block_id,
                w.dst_port.port.name,
            )
            wire_power[key] = -math.inf

        # Find sources and seed
        queue: List[Tuple] = []
        for bid, item in self._block_items.items():
            if isinstance(item.block, Source):
                if item.block.comment_mode == "out":
                    continue
                sig = item.block.generate()
                signals_at.setdefault(bid, {})[item.block.output_ports[0].name] = sig
                queue.append((bid, item.block.output_ports[0].name, sig))
            else:
                item.set_power_warning("ok")

        # Event-based propagation
        max_iterations = max(MIN_PROPAGATION_ITERATIONS, len(self._connections) * ITERATIONS_PER_CONNECTION)
        iterations = 0
        while queue:
            iterations += 1
            if iterations > max_iterations:
                break
            src_bid, src_port, sig = queue.pop(0)
            for dst_bid, dst_port in adj.get((src_bid, src_port), []):
                dst_item = self._block_items.get(dst_bid)
                if dst_item is None:
                    continue
                c_key = (src_bid, src_port, dst_bid, dst_port)
                wire_power[c_key] = sig.power_dbm
                merged_in = self._merge_signals(signals_at.setdefault(dst_bid, {}).get(dst_port), sig)
                prev_in = signals_at.setdefault(dst_bid, {}).get(dst_port)
                signals_at.setdefault(dst_bid, {})[dst_port] = merged_in

                # Power warning
                status = dst_item.block.check_power(merged_in.power_dbm)
                dst_item.set_power_warning(status)

                if self._signals_equivalent(prev_in, merged_in):
                    continue

                if dst_item.block.comment_mode == "out":
                    continue

                if dst_item.block.comment_mode == "through":
                    result = {p.name: merged_in.copy() for p in dst_item.block.output_ports}
                else:
                    result = dst_item.block.process(merged_in, dst_port)
                    for out_sig in result.values():
                        in_noise_floor = merged_in.get_noise_floor_dbm()
                        if in_noise_floor is not None:
                            effective_gain = out_sig.power_dbm - merged_in.power_dbm
                            out_noise_floor = in_noise_floor + effective_gain + max(0.0, dst_item.block.nf_db)
                            out_sig.set_noise_floor_dbm(out_noise_floor)
                        elif merged_in.snr_db is not None and out_sig.snr_db is None:
                            out_sig.snr_db = merged_in.snr_db - max(0.0, dst_item.block.nf_db)

                for out_port, out_sig in result.items():
                    prev_out = signals_at.setdefault(dst_bid, {}).get(out_port)
                    if self._signals_equivalent(prev_out, out_sig):
                        continue
                    signals_at.setdefault(dst_bid, {})[out_port] = out_sig
                    queue.append((dst_bid, out_port, out_sig))

        for w in self._wires:
            key = (
                w.src_port.parentItem().block.block_id,
                w.src_port.port.name,
                w.dst_port.parentItem().block.block_id,
                w.dst_port.port.name,
            )
            pwr = wire_power.get(key, -math.inf)
            dst_block = w.dst_port.parentItem().block
            color = QColor("#BBBBFF")
            if math.isfinite(pwr):
                if dst_block.max_input_power_dbm is not None and pwr > dst_block.max_input_power_dbm:
                    color = QColor("#FF3333")
                else:
                    p1db_in = None
                    if dst_block.p1db_dbm is not None:
                        p1db_in = dst_block.p1db_dbm - dst_block.gain_db
                    if p1db_in is not None and pwr > p1db_in:
                        color = QColor("#FFD75E")
            text = f"{pwr:.2f} dBm" if math.isfinite(pwr) else "-∞ dBm"
            w.set_power_label(text, color)

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

    def get_wires(self) -> List[WireItem]:
        return list(self._wires)

    def clear_scene(self) -> None:
        """Remove all items and reset state."""
        super().clear()
        self._block_items.clear()
        self._wires.clear()
        self._connections.clear()
        self.scene_changed.emit()


# ======================================================================= #
# Canvas View                                                              #
# ======================================================================= #

class RFCanvasView(QGraphicsView):
    """QGraphicsView with zoom/pan support for the RF canvas."""

    def __init__(self, scene: RFScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        # Rubber-band for multi-selection; right-click drag pans manually
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._zoom = 1.0

        # Right-click pan state
        self._panning = False
        self._pan_start = None

        # Touch gestures
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self._touch_start_distance = 0.0
        self._touch_start_zoom = 1.0

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._zoom *= factor
        self._zoom = max(0.1, min(self._zoom, 10.0))
        self.setTransform(QTransform().scale(self._zoom, self._zoom))

    # ------------------------------------------------------------------ #
    # Right-click panning                                                  #
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:
        # Suppress context menu while or just after panning
        if self._panning:
            event.accept()
            return
        super().contextMenuEvent(event)

    def zoom_in(self) -> None:
        """Zoom in."""
        self._zoom = min(self._zoom * 1.2, 10.0)
        self.setTransform(QTransform().scale(self._zoom, self._zoom))

    def zoom_out(self) -> None:
        """Zoom out."""
        self._zoom = max(self._zoom / 1.2, 0.1)
        self.setTransform(QTransform().scale(self._zoom, self._zoom))

    def zoom_reset(self) -> None:
        """Reset zoom to 100%."""
        self._zoom = 1.0
        self.setTransform(QTransform().scale(1.0, 1.0))

    def zoom_to_fit(self) -> None:
        """Zoom and pan to fit all items in view."""
        rect = self.scene().itemsBoundingRect()
        if not rect.isValid():
            self._zoom = 1.0
            self.setTransform(QTransform().scale(1.0, 1.0))
            return
        
        # Add padding
        rect.adjust(-50, -50, 50, 50)
        
        # Calculate zoom to fit
        view_rect = self.viewport().rect()
        scale_x = view_rect.width() / rect.width() if rect.width() > 0 else 1.0
        scale_y = view_rect.height() / rect.height() if rect.height() > 0 else 1.0
        self._zoom = min(scale_x, scale_y, 10.0)
        self._zoom = max(self._zoom, 0.1)
        
        self.setTransform(QTransform().scale(self._zoom, self._zoom))
        self.fitInView(rect, Qt.KeepAspectRatio)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Delete:
            for item in self.scene().selectedItems():
                if isinstance(item, BlockItem):
                    self.scene().remove_block(item.block.block_id)
                elif isinstance(item, AnnotationItem):
                    self.scene().removeItem(item)
                elif isinstance(item, WireItem):
                    self.scene()._remove_wire(item)
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom = min(self._zoom * 1.2, 10.0)
            self.setTransform(QTransform().scale(self._zoom, self._zoom))
        elif event.key() == Qt.Key_Minus:
            self._zoom = max(self._zoom / 1.2, 0.1)
            self.setTransform(QTransform().scale(self._zoom, self._zoom))
        elif event.key() == Qt.Key_0:
            self._zoom = 1.0
            self.setTransform(QTransform().scale(1.0, 1.0))
        elif event.key() == Qt.Key_Home:
            self.zoom_to_fit()
        else:
            super().keyPressEvent(event)

    def touchEvent(self, event) -> bool:
        """Handle touch events for multi-touch gestures (pinch-to-zoom, pan)."""
        from PySide6.QtCore import QEvent
        _TOUCH_BEGIN  = QEvent.TouchBegin
        _TOUCH_UPDATE = QEvent.TouchUpdate

        # PySide6 uses event.points(); fall back to touchPoints() for compatibility
        try:
            touch_points = event.points()
        except AttributeError:
            touch_points = event.touchPoints()

        if len(touch_points) == 2:
            # Two-finger pinch-to-zoom
            p1 = touch_points[0].screenPos()
            p2 = touch_points[1].screenPos()
            distance = math.sqrt((p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2)

            if event.type() == _TOUCH_BEGIN:
                self._touch_start_distance = distance
                self._touch_start_zoom = self._zoom
            elif event.type() == _TOUCH_UPDATE:
                if self._touch_start_distance > 0:
                    scale_factor = distance / self._touch_start_distance
                    new_zoom = self._touch_start_zoom * scale_factor
                    new_zoom = max(0.1, min(new_zoom, 10.0))
                    self._zoom = new_zoom
                    self.setTransform(QTransform().scale(self._zoom, self._zoom))
            return True
        elif len(touch_points) == 1:
            # Single-finger pan
            touch_point = touch_points[0]
            if event.type() == _TOUCH_UPDATE:
                try:
                    delta = touch_point.screenPos() - touch_point.lastScreenPos()
                except AttributeError:
                    delta = touch_point.screenPos() - touch_point.startScreenPos()
                self.horizontalScrollBar().setValue(
                    self.horizontalScrollBar().value() - int(delta.x())
                )
                self.verticalScrollBar().setValue(
                    self.verticalScrollBar().value() - int(delta.y())
                )
            return True
        return False
