"""
Main application window for RF System Tool.
"""
from __future__ import annotations

import os
import json
from typing import Optional, Dict, List
import uuid
import copy

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QStatusBar,
    QFileDialog, QMessageBox, QWidget, QLabel, QMenu, QApplication, QTextEdit,
)
from PySide6.QtGui import QAction, QKeySequence, QBrush, QColor
from PySide6.QtCore import Qt, QPointF, QSettings

from rf_tool.gui.canvas import RFScene, RFCanvasView, WireItem
from rf_tool.gui.node_items import BlockItem, AnnotationItem
from rf_tool.gui.dialogs import PropertiesPanel, CascadeReadoutDialog, SourceSinkMetricsPanel
from rf_tool.gui.ribbon import RibbonWidget
from rf_tool.gui.themes import THEMES
from rf_tool.plots.plot_windows import (
    SpectrumPlot, ActualSpectrumPlot, GainNFPlot, FrequencyResponseView,
    FrequencyComponentEditor, compute_frequency_sweep,
)
from rf_tool.blocks.components import (
    Amplifier, Attenuator, Mixer, SparBlock, TransferFnBlock,
    LowPassFilter, HighPassFilter, PowerSplitter, PowerCombiner, Switch, Source, Sink,
    block_from_dict,
)
from rf_tool.blocks.hierarchical import (
    HierInputPin,
    HierOutputPin,
    HierSubcircuit,
    analysis_blocks_from_subcircuit,
)
from rf_tool.models.rf_block import RFBlock
from rf_tool.engine.cascade import compute_cascade_metrics
from rf_tool.serialization.json_io import save_scene, load_scene
from rf_tool.export.exporters import export_cascade_csv, export_html_report, export_canvas_image

_MAX_RECENT_FILES = 10


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF System Tool")
        self.resize(1200, 760)
        self._current_file: Optional[str] = None
        self._scene_metadata: Dict = {}
        self._p2p_start: Optional[str] = None
        self._p2p_end: Optional[str] = None
        self._child_windows: List[QMainWindow] = []

        self._setup_scene()
        self._setup_dock_properties()
        self._setup_dock_metrics()
        self._setup_dock_messages()
        self._setup_toolbar()
        self._setup_menu()
        self._setup_status_bar()
        self._clipboard_payload: Optional[Dict] = None
        self._paste_count: int = 0

        self._scene.block_selected.connect(self._on_block_selected)
        self._scene.block_double_clicked.connect(self._on_block_double_clicked)
        self._scene.scene_changed.connect(self._on_scene_changed)
        self._scene.wire_selected.connect(self._on_wire_selected)
        self._scene.blocks_selected.connect(self._on_blocks_selected)
        self._refresh_metrics_block_lists()

        # Spectrum viewer
        self._spectrum_plot: Optional[ActualSpectrumPlot] = None

        # Apply default theme
        self._apply_theme("Dark")
        self._refresh_subcircuit_buttons()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
    def _setup_scene(self) -> None:
        self._scene = RFScene()
        self._view = RFCanvasView(self._scene)
        self.setCentralWidget(self._view)

    def _setup_dock_properties(self) -> None:
        dock = QDockWidget("Properties", self)
        dock.setObjectName("propertiesDock")
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._props_panel = PropertiesPanel()
        self._props_panel.block_changed.connect(self._on_block_property_changed)
        self._props_panel.block_ports_changed.connect(self._scene.rebuild_block_ports)
        dock.setWidget(self._props_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._properties_dock = dock

    def _setup_dock_metrics(self) -> None:
        dock = QDockWidget("Source/Sink Metrics", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._metrics_panel = SourceSinkMetricsPanel()
        self._metrics_panel.source_changed.connect(lambda _: self._update_metrics_panel())
        self._metrics_panel.sink_changed.connect(lambda _: self._update_metrics_panel())
        dock.setWidget(self._metrics_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.splitDockWidget(self._properties_dock, dock, Qt.Vertical)

    def _setup_dock_messages(self) -> None:
        dock = QDockWidget("Messages", self)
        dock.setObjectName("messagesDock")
        dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self._messages_console = QTextEdit()
        self._messages_console.setReadOnly(True)
        self._messages_console.setStyleSheet(
            "QTextEdit { background: #0F1218; color: #F2F2F2; font-family: Consolas, Menlo, monospace; }"
        )
        dock.setWidget(self._messages_console)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self._messages_dock = dock

    def _setup_toolbar(self) -> None:
        tb = self.addToolBar("Ribbon")
        tb.setMovable(False)
        tb.setFloatable(False)

        self._ribbon = RibbonWidget(self)

        # Connect ribbon signals to main window handlers
        self._ribbon.sig_zoom_fit.connect(self._zoom_to_fit)
        self._ribbon.sig_propagate.connect(self._propagate_signals)

        # Components
        self._ribbon.sig_add_source.connect(self._add_source)
        self._ribbon.sig_add_sink.connect(self._add_sink)
        self._ribbon.sig_add_amplifier.connect(self._add_amplifier)
        self._ribbon.sig_add_attenuator.connect(self._add_attenuator)
        self._ribbon.sig_add_mixer.connect(self._add_mixer)
        self._ribbon.sig_add_switch.connect(self._add_switch)
        self._ribbon.sig_add_spar.connect(self._add_spar)
        self._ribbon.sig_add_transfer_fn.connect(self._add_transfer_fn)
        self._ribbon.sig_add_lpf.connect(self._add_lpf)
        self._ribbon.sig_add_hpf.connect(self._add_hpf)
        self._ribbon.sig_add_splitter.connect(self._add_splitter)
        self._ribbon.sig_add_combiner.connect(self._add_combiner)
        self._ribbon.sig_add_annotation.connect(self._add_annotation)

        # Analysis
        self._ribbon.sig_p2p_cascade.connect(self._run_p2p_analysis)
        self._ribbon.sig_freq_plot.connect(self._run_freq_plot)
        self._ribbon.sig_signal_spectrum.connect(self._open_spectrum_viewer)
        self._ribbon.sig_freq_response.connect(self._open_frequency_response)

        # Tools
        self._ribbon.sig_cut.connect(self._cut_selected)
        self._ribbon.sig_copy.connect(self._copy_selected)
        self._ribbon.sig_paste.connect(self._paste_selected)
        self._ribbon.sig_comment_out.connect(lambda: self._set_selected_comment_mode("out"))
        self._ribbon.sig_comment_through.connect(lambda: self._set_selected_comment_mode("through"))
        self._ribbon.sig_uncomment.connect(lambda: self._set_selected_comment_mode("active"))
        self._ribbon.sig_select_all.connect(self._scene.select_all)
        self._ribbon.sig_delete_selected.connect(self._delete_selected)

        # Hierarchical
        self._ribbon.sig_add_hier_input.connect(self._add_hier_input_pin)
        self._ribbon.sig_add_hier_output.connect(self._add_hier_output_pin)
        self._ribbon.sig_open_symbol_editor.connect(self._open_symbol_editor)
        self._ribbon.sig_reload_all.connect(self._reload_all_subcircuits)
        self._ribbon.sig_add_subcircuit.connect(self._add_subcircuit)

        tb.addWidget(self._ribbon)
        tb.setFixedHeight(self._ribbon.sizeHint().height())

    def _zoom_to_fit(self) -> None:
        """Fit all items in view."""
        self._view.zoom_to_fit()
        self._status.showMessage("Zoomed to fit")

    # ------------------------------------------------------------------ #
    # Recent files (uses QSettings for platform-standard storage)         #
    # ------------------------------------------------------------------ #
    def _recent_files(self) -> List[str]:
        s = QSettings()
        raw = s.value("recentFiles", [])
        return list(raw) if isinstance(raw, list) else []

    def _add_recent_file(self, path: str) -> None:
        s = QSettings()
        files: List[str] = self._recent_files()
        if path in files:
            files.remove(path)
        files.insert(0, path)
        files = files[:_MAX_RECENT_FILES]
        s.setValue("recentFiles", files)

    def _populate_recent_menu(self) -> None:
        self._recent_menu.clear()
        files = self._recent_files()
        if not files:
            empty_act = QAction("(no recent files)", self)
            empty_act.setEnabled(False)
            self._recent_menu.addAction(empty_act)
            return
        for path in files:
            label = os.path.basename(path)
            act = QAction(label, self)
            act.setToolTip(path)
            act.triggered.connect(lambda _checked, p=path: self._open_scene_path(p))
            self._recent_menu.addAction(act)
        self._recent_menu.addSeparator()
        clear_act = QAction("Clear Recent Files", self)
        clear_act.triggered.connect(self._clear_recent_files)
        self._recent_menu.addAction(clear_act)

    def _clear_recent_files(self) -> None:
        QSettings().setValue("recentFiles", [])

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        act_new = QAction("&New", self, shortcut=QKeySequence.New)
        act_new.triggered.connect(self._new_scene)
        act_open = QAction("&Open…", self, shortcut=QKeySequence.Open)
        act_open.triggered.connect(self._open_scene)
        act_save = QAction("&Save", self, shortcut=QKeySequence.Save)
        act_save.triggered.connect(self._save_scene)
        act_save_as = QAction("Save &As…", self)
        act_save_as.triggered.connect(self._save_scene_as)
        file_menu.addAction(act_new)
        file_menu.addAction(act_open)

        # Open Recent submenu
        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._recent_menu.aboutToShow.connect(self._populate_recent_menu)

        file_menu.addSeparator()
        file_menu.addAction(act_save)
        file_menu.addAction(act_save_as)
        file_menu.addSeparator()

        # Export
        export_menu = file_menu.addMenu("&Export")
        act_csv = QAction("Export CSV…", self)
        act_csv.triggered.connect(self._export_csv)
        act_png = QAction("Export Image (PNG)…", self)
        act_png.triggered.connect(self._export_png)
        act_html = QAction("Export HTML Report…", self)
        act_html.triggered.connect(self._export_html)
        export_menu.addAction(act_csv)
        export_menu.addAction(act_png)
        export_menu.addAction(act_html)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self, shortcut=QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        act_cut = QAction("Cu&t", self, shortcut=QKeySequence.Cut)
        act_cut.triggered.connect(self._cut_selected)
        act_copy = QAction("&Copy", self, shortcut=QKeySequence.Copy)
        act_copy.triggered.connect(self._copy_selected)
        act_paste = QAction("&Paste", self, shortcut=QKeySequence.Paste)
        act_paste.triggered.connect(self._paste_selected)
        act_del = QAction("&Delete Selected", self, shortcut=Qt.Key_Delete)
        act_del.triggered.connect(self._delete_selected)
        act_sel_all = QAction("Select &All", self, shortcut=QKeySequence.SelectAll)
        act_sel_all.triggered.connect(self._scene.select_all)
        act_comment_out = QAction("Comment &Out", self)
        act_comment_out.triggered.connect(lambda: self._set_selected_comment_mode("out"))
        act_comment_through = QAction("Comment T&hrough", self)
        act_comment_through.triggered.connect(lambda: self._set_selected_comment_mode("through"))
        act_uncomment = QAction("&Uncomment", self)
        act_uncomment.triggered.connect(lambda: self._set_selected_comment_mode("active"))
        edit_menu.addAction(act_cut)
        edit_menu.addAction(act_copy)
        edit_menu.addAction(act_paste)
        edit_menu.addSeparator()
        edit_menu.addAction(act_del)
        edit_menu.addAction(act_sel_all)
        edit_menu.addSeparator()
        edit_menu.addAction(act_comment_out)
        edit_menu.addAction(act_comment_through)
        edit_menu.addAction(act_uncomment)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        act_p2p = QAction("&P2P Cascade Readout…", self)
        act_p2p.triggered.connect(self._run_p2p_analysis)
        act_freq = QAction("&Frequency Plot…", self)
        act_freq.triggered.connect(self._run_freq_plot)
        act_prop = QAction("&Propagate Signals", self)
        act_prop.triggered.connect(self._propagate_signals)
        act_spectrum = QAction("&Signal Spectrum Viewer…", self)
        act_spectrum.triggered.connect(self._open_spectrum_viewer)
        act_freq_response = QAction("Frequency &Response…", self)
        act_freq_response.triggered.connect(self._open_frequency_response)
        analysis_menu.addAction(act_p2p)
        analysis_menu.addAction(act_freq)
        analysis_menu.addSeparator()
        analysis_menu.addAction(act_spectrum)
        analysis_menu.addAction(act_freq_response)
        analysis_menu.addSeparator()
        analysis_menu.addAction(act_prop)

        # View menu
        view_menu = menubar.addMenu("&View")
        act_zoom_fit = QAction("Zoom to &Fit", self, shortcut="Home")
        act_zoom_fit.triggered.connect(self._zoom_to_fit)
        act_zoom_in = QAction("Zoom &In", self, shortcut=QKeySequence.ZoomIn)
        act_zoom_in.triggered.connect(lambda: self._view.zoom_in())
        act_zoom_out = QAction("Zoom &Out", self, shortcut=QKeySequence.ZoomOut)
        act_zoom_out.triggered.connect(lambda: self._view.zoom_out())
        act_zoom_reset = QAction("&Reset Zoom", self, shortcut="0")
        act_zoom_reset.triggered.connect(lambda: self._view.zoom_reset())
        view_menu.addAction(act_zoom_fit)
        view_menu.addAction(act_zoom_in)
        view_menu.addAction(act_zoom_out)
        view_menu.addAction(act_zoom_reset)
        view_menu.addSeparator()

        # Themes submenu
        themes_menu = view_menu.addMenu("&Themes")
        for theme_name in THEMES:
            act_theme = QAction(theme_name, self)
            act_theme.triggered.connect(
                lambda checked=False, tn=theme_name: self._apply_theme(tn)
            )
            themes_menu.addAction(act_theme)

    def _setup_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    def _append_runtime_message(self, message: str, level: str = "info") -> None:
        level_colors = {
            "info": "#D0D0D0",
            "warning": "#E8C35E",
            "error": "#FF6B6B",
        }
        color = level_colors.get(level, level_colors["info"])
        safe = (
            message.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self._messages_console.append(f'<span style="color:{color};">{safe}</span>')

    # ------------------------------------------------------------------ #
    # Add block helpers                                                    #
    # ------------------------------------------------------------------ #
    def _next_pos(self) -> QPointF:
        """Return a reasonable drop position in the scene."""
        center = self._view.mapToScene(
            self._view.viewport().rect().center()
        )
        import random
        return QPointF(
            center.x() + random.uniform(-80, 80),
            center.y() + random.uniform(-40, 40),
        )

    def _add_block(self, block) -> None:
        pos = self._next_pos()
        block.x, block.y = pos.x(), pos.y()
        self._scene.add_block(block)
        self._scene.add_block   # type hint hack ignored
        self._status.showMessage(f"Added {block.BLOCK_TYPE}: {block.label}")

    def _add_amplifier(self):  self._add_block(Amplifier())
    def _add_attenuator(self): self._add_block(Attenuator())
    def _add_mixer(self):      self._add_block(Mixer())
    def _add_spar(self):       self._add_block(SparBlock())
    def _add_transfer_fn(self):self._add_block(TransferFnBlock())
    def _add_lpf(self):        self._add_block(LowPassFilter())
    def _add_hpf(self):        self._add_block(HighPassFilter())
    def _add_splitter(self):   self._add_block(PowerSplitter())
    def _add_combiner(self):   self._add_block(PowerCombiner())
    def _add_switch(self):     self._add_block(Switch())
    def _add_source(self):     self._add_block(Source())
    def _add_sink(self):       self._add_block(Sink())

    def _add_annotation(self):
        self._scene.add_annotation("Annotation", *self._next_pos().toTuple())

    # ------------------------------------------------------------------ #
    # Hierarchical block helpers                                           #
    # ------------------------------------------------------------------ #
    def _add_hier_input_pin(self) -> None:
        self._add_block(HierInputPin(pin_name="IN"))

    def _add_hier_output_pin(self) -> None:
        self._add_block(HierOutputPin(pin_name="OUT"))

    def _add_subcircuit(self, path: str) -> None:
        """Place a HierSubcircuit block referencing *path*."""
        block = HierSubcircuit(subcircuit_path=path)
        self._add_block(block)
        if block.file_missing:
            QMessageBox.warning(
                self, "Missing File",
                f"Subcircuit file not found:\n{path}\nBlock placed in MISSING state."
            )

    def _reload_all_subcircuits(self) -> None:
        """Reload port definitions for every HierSubcircuit in the scene."""
        reloaded = 0
        for block in self._scene.get_all_blocks():
            if isinstance(block, HierSubcircuit):
                block.reload()
                item = self._scene.get_block_item(block.block_id)
                if item:
                    item.rebuild_ports()
                    item.update()
                reloaded += 1
        self._status.showMessage(f"Reloaded {reloaded} subcircuit block(s)")

    def _open_symbol_editor(self) -> None:
        """Open the symbol editor for the currently open circuit."""
        from rf_tool.gui.symbol_editor import SymbolEditorDialog
        input_pins = [
            b.pin_name for b in self._scene.get_all_blocks()
            if isinstance(b, HierInputPin)
        ]
        output_pins = [
            b.pin_name for b in self._scene.get_all_blocks()
            if isinstance(b, HierOutputPin)
        ]
        pins = input_pins + output_pins
        if not pins:
            QMessageBox.information(self, "Symbol Editor",
                                    "Add at least one Hierarchical In/Out Pin in this circuit first.")
            return
        initial_symbol = self._scene_metadata.get("symbol", {})
        dlg = SymbolEditorDialog(pins=pins, initial_symbol=initial_symbol, parent=self)
        if dlg.exec():
            self._scene_metadata["symbol"] = dlg.get_symbol()
            self._status.showMessage("Circuit symbol updated")

    def _scan_for_subcircuits(self) -> List[tuple]:
        """
        Scan the directory of the current file for *.json scenes that contain
        HierInputPin or HierOutputPin blocks.

        Returns list of (label, path) tuples.
        """
        if self._current_file:
            directory = os.path.dirname(os.path.abspath(self._current_file))
            current_file = os.path.abspath(self._current_file)
        else:
            directory = os.getcwd()
            current_file = None
        results = []
        try:
            for fname in sorted(os.listdir(directory)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(directory, fname)
                if current_file and os.path.abspath(fpath) == current_file:
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    block_types = {b.get("block_type", "") for b in data.get("blocks", [])}
                    if "HierInputPin" in block_types or "HierOutputPin" in block_types:
                        label = os.path.splitext(fname)[0]
                        results.append((label, fpath))
                except Exception:
                    pass
        except Exception:
            pass
        return results

    def _refresh_subcircuit_buttons(self) -> None:
        """Refresh discovered subcircuit buttons in the ribbon."""
        if hasattr(self, "_ribbon"):
            self._ribbon.refresh_subcircuit_buttons(self._scan_for_subcircuits())

    # ------------------------------------------------------------------ #
    # Theme support                                                        #
    # ------------------------------------------------------------------ #
    def _apply_theme(self, theme_name: str) -> None:
        theme = THEMES.get(theme_name)
        if not theme:
            return
        app = QApplication.instance()
        if app:
            app.setStyleSheet(theme["qss"])
        self._scene.setBackgroundBrush(QBrush(QColor(theme["canvas_bg"])))
        self._status.showMessage(f"Theme applied: {theme_name}")

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #
    def _on_block_selected(self, block_id: str) -> None:
        item = self._scene.get_block_item(block_id)
        if item:
            self._props_panel.set_block(item.block)
            self._status.showMessage(
                f"Selected: {item.block.BLOCK_TYPE} — {item.block.label}"
            )

    def _on_block_property_changed(self, block_id: str) -> None:
        item = self._scene.get_block_item(block_id)
        if item:
            item.update_label()
            item.update()
        self._update_metrics_panel()
        self._status.showMessage("Property updated")

    def _on_scene_changed(self) -> None:
        title = "RF System Tool"
        if self._current_file:
            title += f" — {os.path.basename(self._current_file)}"
        self.setWindowTitle(title + " *")
        self._refresh_metrics_block_lists()
        self._update_metrics_panel()

    def _delete_selected(self) -> None:
        for item in list(self._scene.selectedItems()):
            if isinstance(item, BlockItem):
                self._scene.remove_block(item.block.block_id)
            elif isinstance(item, AnnotationItem):
                self._scene.removeItem(item)
            elif isinstance(item, WireItem):
                self._scene._remove_wire(item)

    def _set_selected_comment_mode(self, mode: str) -> None:
        changed = False
        for item in self._scene.selectedItems():
            if isinstance(item, BlockItem):
                item.block.comment_mode = mode
                item.update()
                changed = True
        if changed:
            self._scene.scene_changed.emit()

    def _copy_selected(self) -> None:
        selected_blocks = [i for i in self._scene.selectedItems() if isinstance(i, BlockItem)]
        selected_ids = {item.block.block_id for item in selected_blocks}
        if not selected_ids:
            self._clipboard_payload = None
            return
        block_dicts = [item.block.to_dict() for item in selected_blocks]
        conns = [
            c for c in self._scene.get_connections()
            if c["src_block_id"] in selected_ids and c["dst_block_id"] in selected_ids
        ]
        annotations = [
            item.to_dict() for item in self._scene.selectedItems()
            if isinstance(item, AnnotationItem)
        ]
        self._clipboard_payload = {"blocks": block_dicts, "connections": conns, "annotations": annotations}
        self._status.showMessage(f"Copied {len(block_dicts)} blocks")

    def _cut_selected(self) -> None:
        self._copy_selected()
        self._delete_selected()
        self._status.showMessage("Cut selection")

    def _paste_selected(self) -> None:
        if not self._clipboard_payload:
            return
        self._paste_count += 1
        offset = 30.0 * self._paste_count
        id_map: Dict[str, str] = {}
        pasted_items = []
        for bd in self._clipboard_payload["blocks"]:
            new_bd = dict(bd)
            old_id = bd["block_id"]
            new_id = str(uuid.uuid4())
            id_map[old_id] = new_id
            new_bd["block_id"] = new_id
            new_bd["x"] = float(bd.get("x", 0.0)) + offset
            new_bd["y"] = float(bd.get("y", 0.0)) + offset
            block = block_from_dict(new_bd)
            pasted_items.append(self._scene.add_block(block))
        for c in self._clipboard_payload["connections"]:
            src_id = id_map.get(c["src_block_id"])
            dst_id = id_map.get(c["dst_block_id"])
            if not src_id or not dst_id:
                continue
            src_item = self._scene.get_block_item(src_id)
            dst_item = self._scene.get_block_item(dst_id)
            if src_item and dst_item:
                self._scene.add_wire(src_item, c["src_port"], dst_item, c["dst_port"])
        for ann in self._clipboard_payload.get("annotations", []):
            self._scene.add_annotation(
                ann.get("text", "Annotation"),
                float(ann.get("x", 0.0)) + offset,
                float(ann.get("y", 0.0)) + offset,
                ann.get("font", "Arial"),
                int(ann.get("font_size", 10)),
                ann.get("color", "#FFFFFF"),
            )
        self._scene.clearSelection()
        for item in pasted_items:
            item.setSelected(True)
        self._status.showMessage(f"Pasted {len(pasted_items)} blocks")

    # ------------------------------------------------------------------ #
    # Propagate signals                                                    #
    # ------------------------------------------------------------------ #
    def _propagate_signals(self) -> None:
        self._append_runtime_message("starting signal propagation.")
        signals = self._scene.propagate_signals(self._append_runtime_message)
        n_blocks = len(signals)
        self._update_metrics_panel()
        self._status.showMessage(f"Propagated signals through {n_blocks} blocks")
        self._append_runtime_message(f"propagation complete across {n_blocks} blocks.")

        # If a sink is selected and has a signal, show spectrum
        selected = self._scene.selectedItems()
        for item in selected:
            if isinstance(item, BlockItem):
                from rf_tool.blocks.components import Sink
                if isinstance(item.block, Sink) and item.block.last_signal:
                    self._open_spectrum(item.block)

    def _open_spectrum(self, sink_block) -> None:
        from rf_tool.plots.plot_windows import SpectrumPlot
        win = SpectrumPlot(None)
        win.set_signal(sink_block.last_signal)
        win.setWindowTitle(f"Spectrum at: {sink_block.label}")
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        win.show()
        # Store reference
        if not hasattr(self, "_plot_windows"):
            self._plot_windows = []
        self._plot_windows.append(win)

    # Handle double-click on Sink from canvas
    def _on_block_double_clicked(self, block_id: str) -> None:
        item = self._scene.get_block_item(block_id)
        if item is None:
            return
        from rf_tool.blocks.components import Sink, Switch, Amplifier, Mixer
        if isinstance(item.block, HierSubcircuit):
            self._open_subcircuit_window(item.block)
        elif isinstance(item.block, Sink):
            self._scene.propagate_signals()
            if item.block.last_signal:
                self._open_spectrum(item.block)
        elif isinstance(item.block, (Amplifier, Mixer)):
            # Open frequency component editor
            dlg = FrequencyComponentEditor(item.block, self)
            dlg.exec()
        elif isinstance(item.block, Switch):
            item.update()  # toggle already done in SwitchItem.mouseDoubleClickEvent

    def _open_subcircuit_window(self, block: HierSubcircuit) -> None:
        """Open a new MainWindow for the given subcircuit block."""
        if block.file_missing or not block.subcircuit_path:
            QMessageBox.warning(self, "Open Subcircuit",
                                f"Subcircuit file not found:\n{block.subcircuit_path}")
            return
        child = MainWindow()
        child.setWindowTitle(f"RF System Tool — {os.path.basename(block.subcircuit_path)}")
        child._open_scene_path(block.subcircuit_path)
        child.show()
        self._child_windows.append(child)

    def _on_wire_selected(self, src_bid: str, src_port: str, dst_bid: str, dst_port: str) -> None:
        """Handle wire selection - show its spectrum in the persistent viewer."""
        signals_at = self._scene.propagate_signals()
        selected_wires = self._scene.get_selected_wires()
        if self._spectrum_plot is None:
            self._spectrum_plot = ActualSpectrumPlot(None)
            self._spectrum_plot.setAttribute(Qt.WA_DeleteOnClose, False)

        if len(selected_wires) > 1:
            overlays = []
            for wire in selected_wires:
                s_bid = wire.src_port.parentItem().block.block_id
                d_bid = wire.dst_port.parentItem().block.block_id
                s_port = wire.src_port.port.name
                d_port = wire.dst_port.port.name
                sig = signals_at.get(d_bid, {}).get(d_port)
                if sig is None:
                    continue
                s_item = self._scene.get_block_item(s_bid)
                d_item = self._scene.get_block_item(d_bid)
                s_lbl = s_item.block.label if s_item else "Source"
                d_lbl = d_item.block.label if d_item else "Dest"
                overlays.append((f"{s_lbl}:{s_port} → {d_lbl}:{d_port}", sig))
            if overlays:
                self._spectrum_plot.set_multi_signals(overlays)
                self._spectrum_plot.set_header("Multi-wire selection", "Signal Spectrum — Multiple Wires")
        else:
            if dst_bid not in signals_at or dst_port not in signals_at[dst_bid]:
                return
            signal = signals_at[dst_bid][dst_port]
            src_item = self._scene.get_block_item(src_bid)
            dst_item = self._scene.get_block_item(dst_bid)
            src_label = src_item.block.label if src_item else "Source"
            dst_label = dst_item.block.label if dst_item else "Dest"
            self._spectrum_plot.set_signal_from_wire(signal, src_label, dst_label, dst_port)
        self._spectrum_plot.show()
        self._spectrum_plot.raise_()
        self._spectrum_plot.activateWindow()

    def _on_blocks_selected(self, block_ids: list) -> None:
        """Handle multi-block selection - overlay all output signals in the spectrum viewer."""
        signals_at = self._scene.propagate_signals()

        signals_with_labels = []
        for bid in block_ids:
            item = self._scene.get_block_item(bid)
            if item is None:
                continue
            block = item.block
            port_signals = signals_at.get(bid, {})
            added_any = False
            for port_name, signal in port_signals.items():
                if signal is None:
                    continue
                label = f"{block.BLOCK_TYPE}: {block.label} [{port_name}]"
                signals_with_labels.append((label, signal))
                added_any = True
            if not added_any:
                label = f"{block.BLOCK_TYPE}: {block.label}"
                signals_with_labels.append((label, None))

        if not any(sig for _, sig in signals_with_labels):
            return

        if self._spectrum_plot is None:
            self._spectrum_plot = ActualSpectrumPlot(None)
            self._spectrum_plot.setAttribute(Qt.WA_DeleteOnClose, False)

        self._spectrum_plot.set_multi_signals(signals_with_labels)
        self._spectrum_plot.set_header("Multi-node selection", "Signal Spectrum — Multiple Nodes")
        self._spectrum_plot.show()
        self._spectrum_plot.raise_()
        self._spectrum_plot.activateWindow()

    def _open_spectrum_viewer(self) -> None:
        """Open the persistent signal spectrum viewer."""
        if self._spectrum_plot is None:
            self._spectrum_plot = ActualSpectrumPlot(None)
            self._spectrum_plot.setAttribute(Qt.WA_DeleteOnClose, False)
        self._spectrum_plot.show()
        self._spectrum_plot.raise_()
        self._spectrum_plot.activateWindow()
        self._status.showMessage("Signal Spectrum Viewer opened - click on wires to show their spectra")

    def _open_frequency_response(self) -> None:
        """Open frequency response viewer with source/sink selection."""
        blocks = self._scene.get_all_blocks()
        if not blocks:
            QMessageBox.information(self, "Frequency Response", "Add blocks first.")
            return
        
        from rf_tool.blocks.components import Source, Sink
        src_rows = [(f"{b.BLOCK_TYPE}: {b.label}", b.block_id) for b in blocks if isinstance(b, Source)]
        sink_rows = [(f"{b.BLOCK_TYPE}: {b.label}", b.block_id) for b in blocks if isinstance(b, Sink)]
        
        if not src_rows or not sink_rows:
            QMessageBox.information(self, "Frequency Response", "Add at least one Source and one Sink.")
            return
        
        win = FrequencyResponseView(None)
        win.set_sources_and_sinks(src_rows, sink_rows)
        
        # Define update callback
        def update_freq_response():
            src_id = win.get_selected_source_id()
            sink_id = win.get_selected_sink_id()
            if src_id and sink_id:
                path_blocks = self._path_blocks(src_id, sink_id)
                if path_blocks:
                    eff = self._effective_blocks(path_blocks)
                    data = compute_frequency_sweep(eff)
                    metrics = compute_cascade_metrics(eff)
                    win.set_data(
                        freq_hz=data["freq_hz"],
                        gain_db=data["gain_db"],
                        nf_db=data["nf_db"],
                        p1db_dbm=metrics.get("p1db_in_dbm"),
                        oip3_dbm=metrics.get("oip3_dbm"),
                    )
        
        win.set_on_selection_changed(update_freq_response)
        update_freq_response()  # Initial update
        
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        win.show()
        if not hasattr(self, "_plot_windows"):
            self._plot_windows = []
        self._plot_windows.append(win)

    # ------------------------------------------------------------------ #
    # P2P analysis                                                         #
    # ------------------------------------------------------------------ #
    def _run_p2p_analysis(self) -> None:
        blocks = self._scene.get_all_blocks()
        if len(blocks) < 2:
            QMessageBox.information(self, "P2P Analysis",
                                    "Add at least two blocks to run cascade analysis.")
            return

        selected_wires = self._scene.get_selected_wires()
        if not selected_wires:
            QMessageBox.information(self, "P2P Analysis", "Select one or more wires on the diagram first.")
            return
        selected_wires.sort(key=lambda w: w.src_port.scenePos().x())
        start_id = selected_wires[0].src_port.parentItem().block.block_id
        end_id = selected_wires[-1].dst_port.parentItem().block.block_id
        labels = {b.block_id: f"{b.BLOCK_TYPE}: {b.label}" for b in blocks}

        path_ids = self._scene.find_path(start_id, end_id)
        if path_ids is None:
            QMessageBox.warning(self, "P2P Analysis",
                                "No connected path found between the selected blocks.")
            return

        path_blocks = [self._scene.get_block_item(bid).block for bid in path_ids
                       if self._scene.get_block_item(bid) is not None]
        eff_blocks = self._effective_blocks(path_blocks)
        metrics = compute_cascade_metrics(eff_blocks)
        stage_labels = [f"{b.BLOCK_TYPE}: {b.label}" for b in eff_blocks]
        self._append_runtime_message(
            f"P2P cascade {labels[start_id]} → {labels[end_id]}: "
            f"Gain {metrics.get('gain_db', 0.0):.2f} dB, NF {metrics.get('nf_db', 0.0):.2f} dB."
        )

        dlg = CascadeReadoutDialog(
            metrics,
            start_label=labels[start_id],
            end_label=labels[end_id],
            stage_labels=stage_labels,
            parent=self,
        )
        dlg.exec()

    def _run_freq_plot(self) -> None:
        blocks = self._scene.get_all_blocks()
        if not blocks:
            QMessageBox.information(self, "Frequency Plot", "Add blocks first.")
            return

        from rf_tool.plots.plot_windows import GainNFPlot, compute_frequency_sweep
        eff_blocks = self._effective_blocks(blocks)
        if not eff_blocks:
            QMessageBox.information(self, "Frequency Plot", "No active blocks to analyze.")
            return
        data = compute_frequency_sweep(eff_blocks)
        metrics = compute_cascade_metrics(eff_blocks)
        self._append_runtime_message(
            f"Frequency plot computed for {len(eff_blocks)} blocks: "
            f"Gain {metrics.get('gain_db', 0.0):.2f} dB, NF {metrics.get('nf_db', 0.0):.2f} dB."
        )

        win = GainNFPlot(None)
        win.set_data(
            freq_hz=data["freq_hz"],
            gain_db=data["gain_db"],
            nf_db=data["nf_db"],
            p1db_dbm=metrics.get("p1db_in_dbm"),
            oip3_dbm=metrics.get("oip3_dbm"),
            damage_dbm=metrics.get("min_damage_dbm"),
            title="Gain / NF vs. Frequency (all blocks)",
        )
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        win.show()
        if not hasattr(self, "_plot_windows"):
            self._plot_windows = []
        self._plot_windows.append(win)

    def _effective_blocks(self, blocks: List[RFBlock]) -> List[RFBlock]:
        out: List[RFBlock] = []
        for b in blocks:
            if b.comment_mode == "out":
                continue
            if isinstance(b, HierSubcircuit):
                out.extend(self._effective_blocks(analysis_blocks_from_subcircuit(b.subcircuit_path)))
                continue
            if b.comment_mode == "through":
                b_passthrough = copy.deepcopy(b)
                b_passthrough.gain_db = 0.0
                b_passthrough.nf_db = 0.0
                b_passthrough.p1db_dbm = None
                b_passthrough.oip3_dbm = None
                out.append(b_passthrough)
            else:
                out.append(b)
        return out

    def _refresh_metrics_block_lists(self) -> None:
        blocks = self._scene.get_all_blocks()
        src_rows = [(f"{b.BLOCK_TYPE}: {b.label}", b.block_id) for b in blocks if isinstance(b, Source)]
        sink_rows = [(f"{b.BLOCK_TYPE}: {b.label}", b.block_id) for b in blocks if isinstance(b, Sink)]
        self._metrics_panel.set_sources(src_rows)
        self._metrics_panel.set_sinks(sink_rows)

    def _path_blocks(self, start_id: Optional[str], end_id: Optional[str]) -> List[RFBlock]:
        if not start_id or not end_id:
            return []
        path = self._scene.find_path(start_id, end_id)
        if not path:
            return []
        return [self._scene.get_block_item(bid).block for bid in path if self._scene.get_block_item(bid)]

    def _max_source_safe_power(self, source_id: Optional[str]) -> Optional[float]:
        if not source_id:
            return None
        sinks = [b for b in self._scene.get_all_blocks() if isinstance(b, Sink)]
        limits: List[float] = []
        for sink in sinks:
            blocks = self._path_blocks(source_id, sink.block_id)
            if not blocks:
                continue
            cum_gain = 0.0
            for b in blocks[1:]:
                if b.comment_mode == "out":
                    break
                if b.max_input_power_dbm is not None:
                    limits.append(b.max_input_power_dbm - cum_gain)
                if b.comment_mode != "through":
                    cum_gain += b.gain_db
        if not limits:
            return None
        return min(limits)

    def _update_metrics_panel(self) -> None:
        source_id = self._metrics_panel.selected_source_id()
        sink_id = self._metrics_panel.selected_sink_id()
        path_blocks = self._path_blocks(source_id, sink_id)
        effective = self._effective_blocks(path_blocks)
        metrics = compute_cascade_metrics(effective) if effective else {}
        source_item = self._scene.get_block_item(source_id) if source_id else None
        sink_level = None
        sink_snr = None
        if source_item and effective:
            src = source_item.block
            if isinstance(src, Source):
                sink_level = src.output_power_dbm + metrics.get("gain_db", 0.0)
                if src.snr_db is not None and metrics.get("nf_db") is not None:
                    sink_snr = src.snr_db - metrics["nf_db"]
        self._metrics_panel.set_metrics(
            sink_level=sink_level,
            sink_snr=sink_snr,
            max_source=self._max_source_safe_power(source_id),
            p1db=metrics.get("p1db_in_dbm") if metrics else None,
            ip3=metrics.get("iip3_dbm") if metrics else None,
        )

    # ------------------------------------------------------------------ #
    # File I/O                                                             #
    # ------------------------------------------------------------------ #
    def _new_scene(self) -> None:
        reply = QMessageBox.question(
            self, "New Scene", "Discard current scene?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._scene.clear_scene()
            self._current_file = None
            self._scene_metadata = {}
            self.setWindowTitle("RF System Tool")
            self._status.showMessage("New scene")
            self._refresh_subcircuit_buttons()

    def _open_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scene", "", "RF Tool Scene (*.json);;All (*)"
        )
        if path:
            self._open_scene_path(path)

    def _open_scene_path(self, path: str) -> None:
        """Load a scene from *path*, updating recent files on success."""
        try:
            data = load_scene(path)
            self._scene.clear_scene()
            for block in data["blocks"]:
                self._scene.add_block(block)
            for conn in data["connections"]:
                src_item = self._scene.get_block_item(conn["src_block_id"])
                dst_item = self._scene.get_block_item(conn["dst_block_id"])
                if src_item and dst_item:
                    self._scene.add_wire(src_item, conn["src_port"],
                                         dst_item, conn["dst_port"])
            for ann in data["annotations"]:
                item = AnnotationItem.from_dict(ann)
                self._scene.addItem(item)
            self._current_file = path
            self._scene_metadata = dict(data.get("metadata", {}))
            self.setWindowTitle(f"RF System Tool — {os.path.basename(path)}")
            self._status.showMessage(f"Loaded {path}")
            self._add_recent_file(path)

            # Warn about missing subcircuit files
            missing = [
                b for b in self._scene.get_all_blocks()
                if isinstance(b, HierSubcircuit) and b.file_missing
            ]
            if missing:
                names = "\n".join(f"  • {b.subcircuit_path}" for b in missing)
                QMessageBox.warning(
                    self, "Missing Subcircuit Files",
                    f"The following subcircuit file(s) were not found:\n{names}\n\n"
                    "Affected blocks are shown in MISSING state."
                )

            # Refresh subcircuit buttons in ribbon
            self._refresh_subcircuit_buttons()

        except Exception as exc:
            QMessageBox.critical(self, "Open Error", str(exc))

    def _save_scene(self) -> None:
        if self._current_file:
            self._do_save(self._current_file)
        else:
            self._save_scene_as()

    def _save_scene_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Scene As", "", "RF Tool Scene (*.json);;All (*)"
        )
        if path:
            self._do_save(path)

    def _do_save(self, path: str) -> None:
        try:
            blocks = self._scene.get_all_blocks()
            connections = self._scene.get_connections()
            annotations = [item.to_dict() for item in self._scene.get_annotations()]
            save_scene(
                blocks,
                connections,
                annotations,
                filepath=path,
                metadata=self._scene_metadata,
            )
            self._current_file = path
            self.setWindowTitle(f"RF System Tool — {os.path.basename(path)}")
            self._status.showMessage(f"Saved {path}")
            self._add_recent_file(path)
            self._refresh_subcircuit_buttons()
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #
    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "cascade_metrics.csv",
            "CSV Files (*.csv);;All (*)"
        )
        if not path:
            return
        try:
            blocks = self._scene.get_all_blocks()
            metrics = compute_cascade_metrics(blocks)
            export_cascade_csv(metrics, blocks, filepath=path)
            self._status.showMessage(f"CSV exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "canvas.png",
            "PNG Images (*.png);;PDF (*.pdf);;All (*)"
        )
        if not path:
            return
        try:
            export_canvas_image(self._scene, filepath=path)
            self._status.showMessage(f"Image exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_html(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export HTML Report", "report.html",
            "HTML Files (*.html);;All (*)"
        )
        if not path:
            return
        try:
            blocks = self._scene.get_all_blocks()
            metrics = compute_cascade_metrics(blocks)
            export_html_report(metrics, blocks, filepath=path)
            self._status.showMessage(f"HTML report exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
