"""
Properties panel and block-editing dialogs.
"""
from __future__ import annotations

from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox,
    QPushButton, QGroupBox, QScrollArea, QColorDialog,
    QDialog, QDialogButtonBox, QFileDialog, QTextEdit,
    QComboBox, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from rf_tool.models.rf_block import RFBlock
from rf_tool.blocks.components import (
    Attenuator, Mixer, SparBlock, TransferFnBlock,
    LowPassFilter, HighPassFilter, PowerSplitter, PowerCombiner, Switch, Source,
)


# ======================================================================= #
# Generic properties panel (dockable widget)                              #
# ======================================================================= #

class PropertiesPanel(QWidget):
    """
    Shows and allows editing of a selected block's properties.
    Emits block_changed when a property is updated.
    """

    block_changed = Signal(str)   # block_id
    block_ports_changed = Signal(str)  # block_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._block: Optional[RFBlock] = None
        self._building = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._title = QLabel("No block selected")
        self._title.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #AAC4FF;"
        )
        layout.addWidget(self._title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(scroll)

        self._form_widget = QWidget()
        self._form_layout = QFormLayout(self._form_widget)
        self._form_layout.setLabelAlignment(Qt.AlignRight)
        scroll.setWidget(self._form_widget)

        layout.addStretch()
        self.setMinimumWidth(220)

    def set_block(self, block: Optional[RFBlock]) -> None:
        self._block = block
        self._rebuild_form()

    def _rebuild_form(self) -> None:
        """Clear and repopulate the form for the current block."""
        self._building = True
        # Remove all form rows
        while self._form_layout.rowCount() > 0:
            self._form_layout.removeRow(0)

        if self._block is None:
            self._title.setText("No block selected")
            self._building = False
            return

        b = self._block
        self._title.setText(f"{b.BLOCK_TYPE}  [{b.block_id[:8]}]")

        self._add_str_row("Label", b.label, self._on_label_changed)
        self._add_enum_row(
            "Comment",
            [("Active", "active"), ("Comment Out", "out"), ("Comment Through", "through")],
            b.comment_mode,
            self._on_comment_mode_changed,
        )

        if not isinstance(b, Source):
            frequency_shaped = isinstance(b, (SparBlock, TransferFnBlock, LowPassFilter, HighPassFilter))
            if not frequency_shaped:
                self._add_float_row("Gain (dB)", b.gain_db, -200, 200, self._on_gain_changed)
            self._add_float_row("NF (dB)", b.nf_db, 0, 100, self._on_nf_changed)
            self._add_optional_float_row("P1dB out (dBm)", b.p1db_dbm, -100, 100, self._on_p1db_changed)
            self._add_optional_float_row("OIP3 (dBm)", b.oip3_dbm, -100, 100, self._on_oip3_changed)
            self._add_optional_float_row("Min Input (dBm)", b.min_input_power_dbm, -200, 100, self._on_min_pwr_changed)
            self._add_optional_float_row("Max Input (dBm)", b.max_input_power_dbm, -200, 100, self._on_max_pwr_changed)

        # Block-specific extras
        if isinstance(b, Attenuator):
            self._add_float_row("Attenuation (dB)", b.attenuation_db, 0, 200,
                                self._on_attenuation_changed)
        elif isinstance(b, SparBlock):
            self._add_file_row("S-param file", b.spar_file or "", self._on_spar_file_changed)
        elif isinstance(b, TransferFnBlock):
            self._add_str_row("Numerator (CSV)", ",".join(str(x) for x in b.numerator),
                              self._on_num_changed)
            self._add_str_row("Denominator (CSV)", ",".join(str(x) for x in b.denominator),
                              self._on_den_changed)
        elif isinstance(b, LowPassFilter) or isinstance(b, HighPassFilter):
            self._add_int_row("Order", b.order, 1, 20, self._on_order_changed)
            self._add_freq_row("Cutoff (Hz)", b.cutoff_hz, self._on_cutoff_changed)
        elif isinstance(b, (PowerSplitter, PowerCombiner)):
            self._add_int_row("N ways", b.n_ways, 2, 16, self._on_nways_changed)
        elif isinstance(b, Source):
            self._add_freq_row("Frequency (Hz)", b.frequency, self._on_src_freq_changed)
            self._add_float_row("Output Power (dBm)", b.output_power_dbm, -200, 100,
                                self._on_src_pwr_changed)
            self._add_optional_float_row("SNR (dB)", b.snr_db, -100, 200, self._on_src_snr_changed)

        # Color picker button
        btn_color = QPushButton("Choose Colour…")
        btn_color.clicked.connect(self._on_choose_color)
        self._form_layout.addRow("", btn_color)

        self._building = False

    # ------------------------------------------------------------------ #
    # Row helpers                                                          #
    # ------------------------------------------------------------------ #
    def _add_str_row(self, label: str, value: str, callback) -> QLineEdit:
        edit = QLineEdit(value)
        edit.textChanged.connect(callback)
        self._form_layout.addRow(label + ":", edit)
        return edit

    def _add_float_row(self, label: str, value: float, min_: float, max_: float, callback) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_, max_)
        spin.setDecimals(2)
        spin.setValue(value)
        spin.valueChanged.connect(callback)
        self._form_layout.addRow(label + ":", spin)
        return spin

    def _add_optional_float_row(self, label: str, value: Optional[float], min_: float, max_: float, callback):
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        cb = QCheckBox()
        cb.setChecked(value is not None)
        spin = QDoubleSpinBox()
        spin.setRange(min_, max_)
        spin.setDecimals(1)
        spin.setValue(value if value is not None else 0.0)
        spin.setEnabled(value is not None)
        h.addWidget(cb)
        h.addWidget(spin)

        def on_toggle(state):
            spin.setEnabled(bool(state))
            if bool(state):
                callback(spin.value())
            else:
                callback(None)

        cb.stateChanged.connect(on_toggle)
        spin.valueChanged.connect(lambda v: callback(v) if cb.isChecked() else None)
        self._form_layout.addRow(label + ":", container)

    def _add_int_row(self, label: str, value: int, min_: int, max_: int, callback) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(min_, max_)
        spin.setValue(value)
        spin.valueChanged.connect(callback)
        self._form_layout.addRow(label + ":", spin)
        return spin

    def _add_enum_row(self, label: str, choices: list, selected_value: str, callback) -> QComboBox:
        combo = QComboBox()
        for display, value in choices:
            combo.addItem(display, value)
        idx = combo.findData(selected_value)
        combo.setCurrentIndex(0 if idx < 0 else idx)
        combo.currentIndexChanged.connect(lambda _: callback(combo.currentData()))
        self._form_layout.addRow(label + ":", combo)
        return combo

    def _add_freq_row(self, label: str, value: float, callback) -> QLineEdit:
        edit = QLineEdit(f"{value:.6g}")
        edit.textChanged.connect(callback)
        self._form_layout.addRow(label + ":", edit)
        return edit

    def _add_file_row(self, label: str, value: str, callback) -> None:
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(value)
        btn = QPushButton("…")
        btn.setMaximumWidth(30)
        h.addWidget(edit)
        h.addWidget(btn)

        def browse():
            path, _ = QFileDialog.getOpenFileName(
                self, "Select S-parameter File", "",
                "Touchstone Files (*.s2p *.s3p *.s4p);;All (*)"
            )
            if path:
                edit.setText(path)

        btn.clicked.connect(browse)
        edit.textChanged.connect(callback)
        self._form_layout.addRow(label + ":", container)

    # ------------------------------------------------------------------ #
    # Change handlers                                                      #
    # ------------------------------------------------------------------ #
    def _emit_changed(self):
        if not self._building and self._block:
            self.block_changed.emit(self._block.block_id)

    def _on_label_changed(self, v): self._block.label = v; self._emit_changed()
    def _on_gain_changed(self, v): self._block.gain_db = v; self._emit_changed()
    def _on_nf_changed(self, v): self._block.nf_db = v; self._emit_changed()
    def _on_p1db_changed(self, v): self._block.p1db_dbm = v; self._emit_changed()
    def _on_oip3_changed(self, v): self._block.oip3_dbm = v; self._emit_changed()
    def _on_min_pwr_changed(self, v): self._block.min_input_power_dbm = v; self._emit_changed()
    def _on_max_pwr_changed(self, v): self._block.max_input_power_dbm = v; self._emit_changed()

    def _on_attenuation_changed(self, v):
        if isinstance(self._block, Attenuator):
            self._block.attenuation_db = abs(v)
            self._block.gain_db = -abs(v)
            self._emit_changed()

    def _on_spar_file_changed(self, v):
        if isinstance(self._block, SparBlock):
            self._block.spar_file = v
            self._block._load_network(v)
            self._emit_changed()

    def _on_num_changed(self, v):
        try:
            if isinstance(self._block, TransferFnBlock):
                self._block.numerator = [float(x) for x in v.split(",")]
                self._emit_changed()
        except ValueError:
            pass

    def _on_den_changed(self, v):
        try:
            if isinstance(self._block, TransferFnBlock):
                self._block.denominator = [float(x) for x in v.split(",")]
                self._emit_changed()
        except ValueError:
            pass

    def _on_order_changed(self, v):
        if hasattr(self._block, "order"):
            self._block.order = v
            self._emit_changed()

    def _on_cutoff_changed(self, v):
        try:
            if hasattr(self._block, "cutoff_hz"):
                self._block.cutoff_hz = float(v)
                self._emit_changed()
        except ValueError:
            pass

    def _on_nways_changed(self, v):
        if isinstance(self._block, (PowerSplitter, PowerCombiner)):
            self._block.set_n_ways(v)
            self.block_ports_changed.emit(self._block.block_id)
            self._emit_changed()

    def _on_src_freq_changed(self, v):
        try:
            if isinstance(self._block, Source):
                self._block.frequency = float(v)
                self._emit_changed()
        except ValueError:
            pass

    def _on_src_pwr_changed(self, v):
        if isinstance(self._block, Source):
            self._block.output_power_dbm = v
            self._emit_changed()

    def _on_src_snr_changed(self, v):
        if isinstance(self._block, Source):
            self._block.snr_db = v
            self._emit_changed()

    def _on_comment_mode_changed(self, v):
        if self._block is not None:
            self._block.comment_mode = v or "active"
            self._emit_changed()

    def _on_choose_color(self):
        if self._block is None:
            return
        color = QColorDialog.getColor(QColor(self._block.color), self)
        if color.isValid():
            self._block.color = color.name()
            self._emit_changed()


# ======================================================================= #
# P2P Cascade Readout Dialog                                              #
# ======================================================================= #

class CascadeReadoutDialog(QDialog):
    """Floating dialog showing P2P cascade metrics."""

    def __init__(self, metrics: dict, start_label: str, end_label: str,
                 stage_labels: Optional[List[str]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("P2P Cascade Analysis")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(f"<b>From:</b> {start_label}  →  <b>To:</b> {end_label}"))

        def fmt(v, suffix="dB"):
            return f"{v:.2f} {suffix}" if v is not None else "N/A"

        form = QFormLayout()
        form.addRow("Total Gain:", QLabel(fmt(metrics.get("gain_db"))))
        form.addRow("Cascaded NF:", QLabel(fmt(metrics.get("nf_db"))))
        form.addRow("System IIP3:", QLabel(fmt(metrics.get("iip3_dbm"), "dBm")))
        form.addRow("System OIP3:", QLabel(fmt(metrics.get("oip3_dbm"), "dBm")))
        form.addRow("Input P1dB:", QLabel(fmt(metrics.get("p1db_in_dbm"), "dBm")))
        form.addRow("Damage Level:", QLabel(fmt(metrics.get("min_damage_dbm"), "dBm")))
        layout.addLayout(form)

        # Stage table
        if metrics.get("stage_gains"):
            layout.addWidget(QLabel("<b>Stage Details:</b>"))
            text = QTextEdit()
            text.setReadOnly(True)
            text.setMaximumHeight(200)
            rows = []
            for i, (g, nf, cg) in enumerate(zip(
                metrics["stage_gains"],
                metrics["stage_nfs"],
                metrics["cumulative_gains"],
            )):
                if stage_labels and i < len(stage_labels):
                    lbl = stage_labels[i]
                else:
                    lbl = f"Stage {i+1}"
                rows.append(f"{lbl}: G={g:.1f} dB, NF={nf:.1f} dB, CumG={cg:.1f} dB")
            text.setPlainText("\n".join(rows))
            layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class SourceSinkMetricsPanel(QWidget):
    """Dock widget content for source/sink-dependent system metrics."""

    source_changed = Signal(str)
    sink_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        form = QFormLayout()
        self._source_combo = QComboBox()
        self._sink_combo = QComboBox()
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._sink_combo.currentIndexChanged.connect(self._on_sink_changed)
        form.addRow("Source:", self._source_combo)
        form.addRow("Sink:", self._sink_combo)
        layout.addLayout(form)

        self._sink_level = QLabel("N/A")
        self._sink_snr = QLabel("N/A")
        self._max_src = QLabel("N/A")
        self._p1db = QLabel("N/A")
        self._ip3 = QLabel("N/A")
        form2 = QFormLayout()
        form2.addRow("Sink level:", self._sink_level)
        form2.addRow("Sink SNR:", self._sink_snr)
        form2.addRow("Max source level (damage):", self._max_src)
        form2.addRow("Cascaded P1dB:", self._p1db)
        form2.addRow("Cascaded IP3:", self._ip3)
        layout.addLayout(form2)
        layout.addStretch()

    def set_sources(self, rows: List[tuple]) -> None:
        current = self.selected_source_id()
        self._source_combo.blockSignals(True)
        self._source_combo.clear()
        for label, block_id in rows:
            self._source_combo.addItem(label, block_id)
        idx = self._source_combo.findData(current)
        if self._source_combo.count() == 0:
            self._source_combo.setCurrentIndex(-1)
        elif idx < 0:
            self._source_combo.setCurrentIndex(0)
        else:
            self._source_combo.setCurrentIndex(idx)
        self._source_combo.blockSignals(False)

    def set_sinks(self, rows: List[tuple]) -> None:
        current = self.selected_sink_id()
        self._sink_combo.blockSignals(True)
        self._sink_combo.clear()
        for label, block_id in rows:
            self._sink_combo.addItem(label, block_id)
        idx = self._sink_combo.findData(current)
        if self._sink_combo.count() == 0:
            self._sink_combo.setCurrentIndex(-1)
        elif idx < 0:
            self._sink_combo.setCurrentIndex(0)
        else:
            self._sink_combo.setCurrentIndex(idx)
        self._sink_combo.blockSignals(False)

    def selected_source_id(self) -> Optional[str]:
        return self._source_combo.currentData()

    def selected_sink_id(self) -> Optional[str]:
        return self._sink_combo.currentData()

    def set_metrics(self, sink_level: Optional[float], sink_snr: Optional[float], max_source: Optional[float],
                    p1db: Optional[float], ip3: Optional[float]) -> None:
        self._sink_level.setText("N/A" if sink_level is None else f"{sink_level:.2f} dBm")
        self._sink_snr.setText("N/A" if sink_snr is None else f"{sink_snr:.2f} dB")
        self._max_src.setText("N/A" if max_source is None else f"{max_source:.2f} dBm")
        self._p1db.setText("N/A" if p1db is None else f"{p1db:.2f} dBm")
        self._ip3.setText("N/A" if ip3 is None else f"{ip3:.2f} dBm")

        self._max_src.setStyleSheet("" if max_source is None else "color: #FF5555; font-weight: bold;")
        self._p1db.setStyleSheet("" if p1db is None else "color: #FFD75E;")
        self._ip3.setStyleSheet("" if ip3 is None else "color: #FFD75E;")

    def _on_source_changed(self):
        source_id = self.selected_source_id()
        if source_id:
            self.source_changed.emit(source_id)

    def _on_sink_changed(self):
        sink_id = self.selected_sink_id()
        if sink_id:
            self.sink_changed.emit(sink_id)
