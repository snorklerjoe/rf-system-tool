"""
Main application window for RF System Tool.
"""
from __future__ import annotations

import os
from typing import Optional, Dict, List

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QToolBar, QStatusBar,
    QFileDialog, QMessageBox, QWidget, QLabel,
    QInputDialog, QApplication,
)
from PySide6.QtGui import QAction, QIcon, QKeySequence, QColor
from PySide6.QtCore import Qt, QPointF

from rf_tool.gui.canvas import RFScene, RFCanvasView
from rf_tool.gui.node_items import BlockItem, AnnotationItem
from rf_tool.gui.dialogs import PropertiesPanel, CascadeReadoutDialog
from rf_tool.blocks.components import (
    Amplifier, Attenuator, Mixer, SparBlock, TransferFnBlock,
    LowPassFilter, HighPassFilter, PowerSplitter, Switch, Source, Sink,
    block_from_dict,
)
from rf_tool.engine.cascade import compute_cascade_metrics
from rf_tool.serialization.json_io import save_scene, load_scene
from rf_tool.export.exporters import export_cascade_csv, export_html_report, export_canvas_image


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF System Tool")
        self.resize(1200, 760)
        self._current_file: Optional[str] = None
        self._p2p_start: Optional[str] = None
        self._p2p_end: Optional[str] = None

        self._setup_scene()
        self._setup_dock_properties()
        self._setup_toolbar()
        self._setup_menu()
        self._setup_status_bar()

        self._scene.block_selected.connect(self._on_block_selected)
        self._scene.scene_changed.connect(self._on_scene_changed)

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
    def _setup_scene(self) -> None:
        self._scene = RFScene()
        self._view = RFCanvasView(self._scene)
        self.setCentralWidget(self._view)

    def _setup_dock_properties(self) -> None:
        dock = QDockWidget("Properties", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._props_panel = PropertiesPanel()
        self._props_panel.block_changed.connect(self._on_block_property_changed)
        dock.setWidget(self._props_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _setup_toolbar(self) -> None:
        tb = self.addToolBar("Blocks")
        tb.setMovable(False)

        block_actions = [
            ("Amplifier",    "▶ Amp",    self._add_amplifier),
            ("Attenuator",   "⬡ Att",    self._add_attenuator),
            ("Mixer",        "✕ Mix",    self._add_mixer),
            ("S-Param",      "[S] Spar", self._add_spar),
            ("H(s)",         "⨍ TF",     self._add_transfer_fn),
            ("LPF",          "⊓ LPF",    self._add_lpf),
            ("HPF",          "⊔ HPF",    self._add_hpf),
            ("Splitter",     "⊕ Spl",    self._add_splitter),
            ("Switch 1×2",   "⇀ SW",     self._add_switch),
            ("Source",       "~ Src",    self._add_source),
            ("Sink",         "⊥ Snk",    self._add_sink),
            (None, None, None),          # separator
            ("Annotate",     "T Ann",    self._add_annotation),
        ]

        for name, label, handler in block_actions:
            if name is None:
                tb.addSeparator()
                continue
            act = QAction(label, self)
            act.setToolTip(f"Add {name}")
            if handler:
                act.triggered.connect(handler)
            tb.addAction(act)

        tb.addSeparator()

        # Analysis actions
        p2p_act = QAction("📐 P2P", self)
        p2p_act.setToolTip("Point-to-Point Cascade Analysis")
        p2p_act.triggered.connect(self._run_p2p_analysis)
        tb.addAction(p2p_act)

        freq_act = QAction("📈 F-Plot", self)
        freq_act.setToolTip("Gain/NF vs Frequency Plot")
        freq_act.triggered.connect(self._run_freq_plot)
        tb.addAction(freq_act)

        prop_act = QAction("⚡ Propagate", self)
        prop_act.setToolTip("Propagate Signals")
        prop_act.triggered.connect(self._propagate_signals)
        tb.addAction(prop_act)

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
        act_del = QAction("&Delete Selected", self, shortcut=Qt.Key_Delete)
        act_del.triggered.connect(self._delete_selected)
        act_sel_all = QAction("Select &All", self, shortcut=QKeySequence.SelectAll)
        act_sel_all.triggered.connect(lambda: self._scene.selectAll())
        edit_menu.addAction(act_del)
        edit_menu.addAction(act_sel_all)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        act_p2p = QAction("&P2P Cascade Readout…", self)
        act_p2p.triggered.connect(self._run_p2p_analysis)
        act_freq = QAction("&Frequency Plot…", self)
        act_freq.triggered.connect(self._run_freq_plot)
        act_prop = QAction("&Propagate Signals", self)
        act_prop.triggered.connect(self._propagate_signals)
        analysis_menu.addAction(act_p2p)
        analysis_menu.addAction(act_freq)
        analysis_menu.addSeparator()
        analysis_menu.addAction(act_prop)

    def _setup_status_bar(self) -> None:
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

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
    def _add_switch(self):     self._add_block(Switch())
    def _add_source(self):     self._add_block(Source())
    def _add_sink(self):       self._add_block(Sink())

    def _add_annotation(self):
        self._scene.add_annotation("Annotation", *self._next_pos().toTuple())

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
        self._status.showMessage("Property updated")

    def _on_scene_changed(self) -> None:
        title = "RF System Tool"
        if self._current_file:
            title += f" — {os.path.basename(self._current_file)}"
        self.setWindowTitle(title + " *")

    def _delete_selected(self) -> None:
        for item in list(self._scene.selectedItems()):
            if isinstance(item, BlockItem):
                self._scene.remove_block(item.block.block_id)
            elif isinstance(item, AnnotationItem):
                self._scene.removeItem(item)

    # ------------------------------------------------------------------ #
    # Propagate signals                                                    #
    # ------------------------------------------------------------------ #
    def _propagate_signals(self) -> None:
        signals = self._scene.propagate_signals()
        n_blocks = len(signals)
        self._status.showMessage(f"Propagated signals through {n_blocks} blocks")

        # If a sink is selected and has a signal, show spectrum
        selected = self._scene.selectedItems()
        for item in selected:
            if isinstance(item, BlockItem):
                from rf_tool.blocks.components import Sink
                if isinstance(item.block, Sink) and item.block.last_signal:
                    self._open_spectrum(item.block)

    def _open_spectrum(self, sink_block) -> None:
        from rf_tool.plots.plot_windows import SpectrumPlot
        win = SpectrumPlot(self)
        win.set_signal(sink_block.last_signal)
        win.setWindowTitle(f"Spectrum at: {sink_block.label}")
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
        from rf_tool.blocks.components import Sink, Switch
        if isinstance(item.block, Sink):
            self._scene.propagate_signals()
            if item.block.last_signal:
                self._open_spectrum(item.block)
        elif isinstance(item.block, Switch):
            item.update()  # toggle already done in SwitchItem.mouseDoubleClickEvent

    # ------------------------------------------------------------------ #
    # P2P analysis                                                         #
    # ------------------------------------------------------------------ #
    def _run_p2p_analysis(self) -> None:
        blocks = self._scene.get_all_blocks()
        if len(blocks) < 2:
            QMessageBox.information(self, "P2P Analysis",
                                    "Add at least two blocks to run cascade analysis.")
            return

        labels = {b.block_id: f"{b.BLOCK_TYPE}: {b.label}" for b in blocks}
        choices = list(labels.values())

        start_choice, ok1 = QInputDialog.getItem(
            self, "P2P Analysis", "Select Start Block:", choices, 0, False)
        if not ok1:
            return
        end_choice, ok2 = QInputDialog.getItem(
            self, "P2P Analysis", "Select End Block:", choices, 1, False)
        if not ok2:
            return

        start_id = [bid for bid, lbl in labels.items() if lbl == start_choice][0]
        end_id = [bid for bid, lbl in labels.items() if lbl == end_choice][0]

        path_ids = self._scene.find_path(start_id, end_id)
        if path_ids is None:
            QMessageBox.warning(self, "P2P Analysis",
                                "No connected path found between the selected blocks.")
            return

        path_blocks = [self._scene.get_block_item(bid).block for bid in path_ids
                       if self._scene.get_block_item(bid) is not None]
        metrics = compute_cascade_metrics(path_blocks)

        dlg = CascadeReadoutDialog(
            metrics,
            start_label=labels[start_id],
            end_label=labels[end_id],
            parent=self,
        )
        dlg.exec()

    def _run_freq_plot(self) -> None:
        blocks = self._scene.get_all_blocks()
        if not blocks:
            QMessageBox.information(self, "Frequency Plot", "Add blocks first.")
            return

        from rf_tool.plots.plot_windows import GainNFPlot, compute_frequency_sweep
        data = compute_frequency_sweep(blocks)
        metrics = compute_cascade_metrics(blocks)

        win = GainNFPlot(self)
        win.set_data(
            freq_hz=data["freq_hz"],
            gain_db=data["gain_db"],
            nf_db=data["nf_db"],
            p1db_dbm=metrics.get("p1db_in_dbm"),
            oip3_dbm=metrics.get("oip3_dbm"),
            damage_dbm=metrics.get("min_damage_dbm"),
            title="Gain / NF vs. Frequency (all blocks)",
        )
        win.show()
        if not hasattr(self, "_plot_windows"):
            self._plot_windows = []
        self._plot_windows.append(win)

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
            self.setWindowTitle("RF System Tool")
            self._status.showMessage("New scene")

    def _open_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scene", "", "RF Tool Scene (*.json);;All (*)"
        )
        if not path:
            return
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
            self.setWindowTitle(f"RF System Tool — {os.path.basename(path)}")
            self._status.showMessage(f"Loaded {path}")
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
            save_scene(blocks, connections, annotations, filepath=path)
            self._current_file = path
            self.setWindowTitle(f"RF System Tool — {os.path.basename(path)}")
            self._status.showMessage(f"Saved {path}")
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
