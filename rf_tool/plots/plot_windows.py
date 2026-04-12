"""
pyqtgraph-based plot windows for RF System Tool.

SpectrumPlot   - Shows frequency-domain spectrum at a Sink node
ActualSpectrumPlot - Persistent spectrum viewer for wire signals with enhanced rendering
GainNFPlot     - Shows Gain / NF vs. frequency for a P2P path
FrequencyResponseView - Frequency response with source/sink selection
"""
from __future__ import annotations

import math
import logging
from typing import List, Optional, Dict, Tuple, Any

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox,
    QComboBox, QTableWidget, QTableWidgetItem, QSpinBox,
    QDoubleSpinBox, QPushButton, QTabWidget, QDialog, QMessageBox,
    QHeaderView, QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal as QtSignal
from PySide6.QtGui import QColor

import pyqtgraph as pg
pg.setConfigOption("background", "k")
pg.setConfigOption("foreground", "w")
logger = logging.getLogger(__name__)

# Colour palette used when displaying multiple overlaid signals
_SIGNAL_COLORS: List[Tuple[int, int, int]] = [
    (0, 200, 255),    # cyan
    (255, 150, 0),    # orange
    (50, 255, 50),    # green
    (255, 255, 0),    # yellow
    (200, 50, 255),   # purple
    (255, 80, 80),    # red
    (50, 200, 150),   # teal
    (255, 150, 200),  # pink
]

_NOISE_COLOR = (80, 80, 80)       # dark grey for noise trace
_NOISE_FLOOR_LINE_COLOR = (140, 140, 140)  # slightly brighter for the reference line


def _draw_noise_floor(plot_widget: pg.PlotWidget,
                      noise_floor_dbm: float,
                      x_min: float, x_max: float) -> None:
    """Draw a noisy noise-floor band across the given x range."""
    n_pts = 600
    xs = np.linspace(x_min, x_max, n_pts)
    ys = noise_floor_dbm + np.random.uniform(-0.5, 0.5, n_pts)
    noise_trace = pg.PlotDataItem(
        xs, ys,
        pen=pg.mkPen(_NOISE_COLOR, width=1.0),
        name="Noise floor",
    )
    plot_widget.addItem(noise_trace)
    # Dashed reference line at the exact noise floor level
    ref_line = pg.InfiniteLine(
        pos=noise_floor_dbm, angle=0,
        pen=pg.mkPen(_NOISE_FLOOR_LINE_COLOR, width=1.0, style=Qt.DashLine),
    )
    plot_widget.addItem(ref_line)


# ======================================================================= #
# Spectrum Plot - Enhanced                                                 #
# ======================================================================= #

class SpectrumPlot(QWidget):
    """
    Shows the frequency-domain spectrum at a Sink node.

    Draws vertical impulse lines for each tone (carrier + spurs) with improved rendering.
    Noise floor is shown as a noisy trace at the bottom of the display.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrum Viewer")
        self.resize(900, 500)
        layout = QVBoxLayout(self)

        # Toolbar row: frequency range controls
        btn_row = QHBoxLayout()
        btn_row.addWidget(QLabel("Start (GHz):"))
        self._start_freq = QDoubleSpinBox()
        self._start_freq.setDecimals(6)
        self._start_freq.setRange(-1e6, 1e6)
        self._start_freq.setSingleStep(0.001)
        self._start_freq.valueChanged.connect(self._apply_manual_range)
        btn_row.addWidget(self._start_freq)
        btn_row.addWidget(QLabel("Stop (GHz):"))
        self._stop_freq = QDoubleSpinBox()
        self._stop_freq.setDecimals(6)
        self._stop_freq.setRange(-1e6, 1e6)
        self._stop_freq.setSingleStep(0.001)
        self._stop_freq.valueChanged.connect(self._apply_manual_range)
        btn_row.addWidget(self._stop_freq)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._plot_widget = pg.PlotWidget(title="Spectrum at Node")
        self._plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self._plot_widget.setLabel("left", "Power", units="dBm")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.addLegend()
        layout.addWidget(self._plot_widget)

        self._info_label = QLabel("")
        self._info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._info_label)

        self._current_signal = None
        self._y_min = None
        self._impulse_floor = None
        self._full_x_range: Optional[Tuple[float, float]] = None

    def _set_range_controls_ghz(self, start_hz: float, stop_hz: float) -> None:
        self._start_freq.blockSignals(True)
        self._stop_freq.blockSignals(True)
        self._start_freq.setValue(start_hz / 1e9)
        self._stop_freq.setValue(stop_hz / 1e9)
        self._start_freq.blockSignals(False)
        self._stop_freq.blockSignals(False)

    def _apply_manual_range(self, *_args) -> None:
        if self._current_signal is not None:
            self.set_signal(self._current_signal)

    def set_signal(self, signal) -> None:
        """
        Populate the plot from a Signal object.

        Parameters
        ----------
        signal : Signal
        """
        self._current_signal = signal
        self._plot_widget.clear()

        fc = signal.carrier_frequency
        pw = signal.power_dbm

        all_freqs = [fc] + [s.frequency for s in signal.spurs]
        all_powers = [pw] + [s.power_dbm for s in signal.spurs]
        self._full_x_range = (min(all_freqs), max(all_freqs))
        noise_floor = signal.get_noise_floor_dbm()

        start_hz = self._start_freq.value() * 1e9
        stop_hz = self._stop_freq.value() * 1e9
        if stop_hz <= start_hz:
            if self._full_x_range is not None and self._start_freq.value() == 0.0 and self._stop_freq.value() == 0.0:
                start_hz, stop_hz = self._full_x_range
                self._set_range_controls_ghz(start_hz, stop_hz)
            else:
                self._plot_widget.setTitle("Spectrum — Invalid frequency range")
                self._info_label.setText("Set Stop frequency greater than Start frequency.")
                return

        in_range_components: List[Tuple[float, float, str]] = []
        if start_hz <= fc <= stop_hz:
            in_range_components.append((fc, pw, "Carrier"))
        for i, spur in enumerate(signal.spurs):
            if start_hz <= spur.frequency <= stop_hz:
                in_range_components.append((spur.frequency, spur.power_dbm, f"Spur {i+1}"))

        if not in_range_components:
            self._plot_widget.setTitle("Spectrum — No tones in selected frequency range")
            self._info_label.setText("No tones in selected start/stop range.")
            if noise_floor is not None:
                _draw_noise_floor(self._plot_widget, noise_floor, start_hz, stop_hz)
            self._plot_widget.setXRange(start_hz, stop_hz, padding=0.0)
            return

        filtered_powers = [p for _, p, _ in in_range_components]
        filtered_freqs = [f for f, _, _ in in_range_components]
        max_power = max(filtered_powers)
        min_power = min(filtered_powers)

        # Noise floor must always be the bottom of the display
        if noise_floor is not None:
            y_min = noise_floor - 5.0
            self._impulse_floor = noise_floor
        else:
            y_min = min_power - 10.0
            self._impulse_floor = y_min
        y_max = max_power + max(10.0, abs(max_power - y_min) * 0.1)
        self._y_min = y_min

        x_draw_min = start_hz
        x_draw_max = stop_hz
        x_draw_span = max(x_draw_max - x_draw_min, 1e6)

        # Draw noise floor trace spanning the full x range
        if noise_floor is not None:
            _draw_noise_floor(self._plot_widget, noise_floor, x_draw_min, x_draw_max)

        # Draw carrier impulse
        if start_hz <= fc <= stop_hz:
            self._draw_impulse(fc, pw, color=(0, 200, 255), name="Carrier",
                               tooltip=f"{fc/1e9:.4f} GHz", x_span=x_draw_span)

        # Draw spurs
        for i, spur in enumerate(signal.spurs):
            if start_hz <= spur.frequency <= stop_hz:
                self._draw_impulse(spur.frequency, spur.power_dbm, color=(255, 100, 0),
                                   name=f"Spur {i+1}", tooltip=f"{spur.frequency/1e9:.4f} GHz",
                                   x_span=x_draw_span)

        # Explicitly set y range so noise floor is always at the bottom
        self._plot_widget.setYRange(y_min, y_max, padding=0)
        self._plot_widget.setXRange(x_draw_min, x_draw_max, padding=0.0)

        self._plot_widget.setTitle(
            f"Spectrum — Carrier: {fc/1e9:.4f} GHz @ {pw:.1f} dBm, Spurs: {len(signal.spurs)}"
        )
        self._info_label.setText(
            (
                f"Carrier: {fc/1e9:.4f} GHz @ {pw:.1f} dBm | "
                f"Noise floor: {noise_floor:.1f} dBm | Spurs: {len(signal.spurs)}"
                if noise_floor is not None else
                f"Carrier: {fc/1e9:.4f} GHz @ {pw:.1f} dBm | Spurs: {len(signal.spurs)}"
            )
        )

    def _draw_impulse(self, freq: float, power_dbm: float, color: Tuple, name: str,
                      tooltip: str = "", x_span: float = 1e6) -> None:
        """Draw an impulse as an inverted-T stem (vertical line + base cap)."""
        bottom = self._impulse_floor if self._impulse_floor is not None else power_dbm - 60

        stem = pg.PlotDataItem(
            [freq, freq], [bottom, power_dbm],
            pen=pg.mkPen(color, width=2.5),
            name=name,
        )
        self._plot_widget.addItem(stem)

        cap_half_width = max(x_span * 0.0025, 1.0)
        base_cap = pg.PlotDataItem(
            [freq - cap_half_width, freq + cap_half_width], [bottom, bottom],
            pen=pg.mkPen(color, width=2.5),
        )
        self._plot_widget.addItem(base_cap)

        dot = pg.ScatterPlotItem(
            x=[freq], y=[power_dbm],
            size=6, brush=pg.mkBrush(color), pen=pg.mkPen(color),
            hoverable=True, hoverSize=8,
        )
        dot.setToolTip(f"{name}: {power_dbm:.1f} dBm @ {tooltip if tooltip else freq:.2e} Hz")
        self._plot_widget.addItem(dot)


# ======================================================================= #
# Actual Spectrum Plot - For Wire Signals / Multi-Node                    #
# ======================================================================= #

class ActualSpectrumPlot(QWidget):
    """
    Persistent spectrum viewer that shows actual signals propagating through wires
    or at selected blocks.

    Supports overlaying multiple signals (ctrl-click multi-node selection) with
    distinct colours and a legend identifying each component.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Signal Spectrum Viewer")
        self.resize(900, 500)
        layout = QVBoxLayout(self)

        # Top bar: wire/node label + range controls
        info_layout = QHBoxLayout()
        self._wire_label = QLabel("Click on a wire to show its spectrum")
        self._wire_label.setStyleSheet("color: #AABBDD; font-weight: bold;")
        info_layout.addWidget(self._wire_label)
        info_layout.addStretch()
        info_layout.addWidget(QLabel("Start (GHz):"))
        self._start_freq = QDoubleSpinBox()
        self._start_freq.setDecimals(6)
        self._start_freq.setRange(-1e6, 1e6)
        self._start_freq.setSingleStep(0.001)
        self._start_freq.valueChanged.connect(self._apply_manual_range)
        info_layout.addWidget(self._start_freq)
        info_layout.addWidget(QLabel("Stop (GHz):"))
        self._stop_freq = QDoubleSpinBox()
        self._stop_freq.setDecimals(6)
        self._stop_freq.setRange(-1e6, 1e6)
        self._stop_freq.setSingleStep(0.001)
        self._stop_freq.valueChanged.connect(self._apply_manual_range)
        info_layout.addWidget(self._stop_freq)
        layout.addLayout(info_layout)

        # Plot widget
        self._plot_widget = pg.PlotWidget(title="Signal Spectrum")
        self._plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self._plot_widget.setLabel("left", "Power", units="dBm")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._legend = self._plot_widget.addLegend(offset=(10, 10))
        layout.addWidget(self._plot_widget)

        # Status info
        self._info_label = QLabel("")
        self._info_label.setAlignment(Qt.AlignCenter)
        self._info_label.setStyleSheet("color: #CCCCCC; font-size: 10pt;")
        layout.addWidget(self._info_label)

        self._current_signal = None
        self._current_wire_info = None
        self._all_signals_with_labels: List[Tuple[str, Any]] = []
        self._full_x_range: Optional[Tuple[float, float]] = None
        self._y_min = None
        self._impulse_floor = None

    def _set_range_controls_ghz(self, start_hz: float, stop_hz: float) -> None:
        self._start_freq.blockSignals(True)
        self._stop_freq.blockSignals(True)
        self._start_freq.setValue(start_hz / 1e9)
        self._stop_freq.setValue(stop_hz / 1e9)
        self._start_freq.blockSignals(False)
        self._stop_freq.blockSignals(False)

    def _apply_manual_range(self, *_args) -> None:
        self._render_multi_signals()

    def set_signal_from_wire(self, signal, src_block_label: str = "",
                             dst_block_label: str = "", port_name: str = "") -> None:
        """
        Display a single signal from a wire connection.

        Parameters
        ----------
        signal : Signal
        src_block_label : str
        dst_block_label : str
        port_name : str
        """
        label = f"{src_block_label} → {dst_block_label}"
        self.set_multi_signals([(label, signal)])
        if signal is not None:
            wire_desc = f"{src_block_label} → {dst_block_label} ({port_name})"
            self._wire_label.setText(wire_desc)
            self._plot_widget.setTitle(
                f"Spectrum at {wire_desc}"
            )
        else:
            self._wire_label.setText("No signal on this wire")
            self._info_label.setText("")

    def set_multi_signals(
        self,
        signals_with_labels: List[Tuple[str, Any]],
    ) -> None:
        self._all_signals_with_labels = list(signals_with_labels)
        self._render_multi_signals()

    def set_header(self, header_text: str, title_text: str) -> None:
        """Set header label and plot title without exposing internals."""
        self._wire_label.setText(header_text)
        self._plot_widget.setTitle(title_text)

    def _render_multi_signals(self) -> None:
        """
        Overlay multiple signals on the same plot with distinct colours.

        Parameters
        ----------
        signals_with_labels : list of (label, signal) tuples
            Each *signal* is a Signal object (or None to skip).
        """
        self._plot_widget.clear()
        # Re-attach legend after clear (clear() removes it)
        self._legend = self._plot_widget.addLegend(offset=(10, 10))

        valid = [(lbl, sig) for lbl, sig in self._all_signals_with_labels if sig is not None]
        if not valid:
            self._info_label.setText("")
            self._full_x_range = None
            return

        # ---- Compute full frequency extent ----
        all_freqs: List[float] = []
        for _lbl, sig in valid:
            all_freqs.append(sig.carrier_frequency)
            all_freqs.extend(s.frequency for s in sig.spurs)
        self._full_x_range = (min(all_freqs), max(all_freqs))

        start_hz = self._start_freq.value() * 1e9
        stop_hz = self._stop_freq.value() * 1e9
        if stop_hz <= start_hz:
            if self._full_x_range is not None and self._start_freq.value() == 0.0 and self._stop_freq.value() == 0.0:
                start_hz, stop_hz = self._full_x_range
                self._set_range_controls_ghz(start_hz, stop_hz)
            else:
                self._plot_widget.setTitle("Signal Spectrum — Invalid frequency range")
                self._info_label.setText("Set Stop frequency greater than Start frequency.")
                return

        # ---- Determine axis ranges across all displayed signals ----
        all_powers: List[float] = []
        all_noise_floors: List[float] = []
        displayed_components: List[Tuple[str, Any, List[Tuple[float, float, bool]]]] = []
        for _lbl, sig in valid:
            comps: List[Tuple[float, float, bool]] = []
            if start_hz <= sig.carrier_frequency <= stop_hz:
                comps.append((sig.carrier_frequency, sig.power_dbm, True))
            for spur in sig.spurs:
                if start_hz <= spur.frequency <= stop_hz:
                    comps.append((spur.frequency, spur.power_dbm, False))
            if not comps:
                continue
            displayed_components.append((_lbl, sig, comps))
            all_powers.extend(power_dbm for _, power_dbm, _ in comps)
            nf = sig.get_noise_floor_dbm()
            if nf is not None:
                all_noise_floors.append(nf)

        if not displayed_components:
            self._plot_widget.setTitle("Signal Spectrum — No tones in selected frequency range")
            self._info_label.setText("No tones in selected start/stop range.")
            min_nf = min(all_noise_floors) if all_noise_floors else None
            if min_nf is not None:
                _draw_noise_floor(self._plot_widget, min_nf, start_hz, stop_hz)
            self._plot_widget.setXRange(start_hz, stop_hz, padding=0.0)
            return

        max_power = max(all_powers)
        min_power = min(all_powers)
        noise_floor = min(all_noise_floors) if all_noise_floors else None

        # Noise floor is always the lowest thing on the display
        if noise_floor is not None:
            y_min = noise_floor - 5.0
            self._impulse_floor = noise_floor
        else:
            y_min = min_power - 10.0
            self._impulse_floor = y_min
        y_max = max_power + max(10.0, abs(max_power - y_min) * 0.1)
        self._y_min = y_min

        x_draw_min = start_hz
        x_draw_max = stop_hz
        x_draw_span = max(x_draw_max - x_draw_min, 1e6)

        # ---- Noise floor trace (drawn first, behind signals) ----
        if noise_floor is not None:
            _draw_noise_floor(self._plot_widget, noise_floor, x_draw_min, x_draw_max)

        # ---- Draw each signal with its own colour ----
        info_parts: List[str] = []
        for i, (label, sig, comps) in enumerate(displayed_components):
            color = _SIGNAL_COLORS[i % len(_SIGNAL_COLORS)]
            sig_nf = sig.get_noise_floor_dbm()
            legend_added = False
            spur_count = 0

            spur_color = tuple(max(0, int(c * 0.65)) for c in color)
            ordered_comps = sorted(comps, key=lambda component: (0 if component[2] else 1, component[0]))
            for (freq_hz, power_dbm, is_carrier) in ordered_comps:
                tone_color = color if is_carrier else spur_color
                # Add exactly one legend entry per selected signal to avoid duplicates.
                legend_name = label if not legend_added else None
                if is_carrier:
                    tone_name = f"{label} carrier"
                else:
                    spur_count += 1
                    tone_name = f"{label} spur {spur_count}"
                # name is for tone-level hover text; legend_name controls one legend row per signal.
                self._draw_impulse(
                    freq_hz,
                    power_dbm,
                    color=tone_color,
                    name=tone_name,
                    is_carrier=is_carrier,
                    tooltip=f"{freq_hz/1e9:.4f} GHz",
                    legend_name=legend_name,
                    x_span=x_draw_span,
                )
                legend_added = True

            strongest = max(comps, key=lambda x: x[1])
            part = f"{label}: {strongest[0]/1e9:.4f} GHz @ {strongest[1]:.1f} dBm"
            if sig_nf is not None:
                part += f" | NF floor {sig_nf:.1f} dBm"
            if sig.snr_db is not None:
                part += f" | SNR {sig.snr_db:.1f} dB"
            info_parts.append(part)

        self._info_label.setText("   ·   ".join(info_parts))

        # ---- Set axis ranges explicitly ----
        self._plot_widget.setYRange(y_min, y_max, padding=0)
        self._plot_widget.setXRange(x_draw_min, x_draw_max, padding=0.0)

        if len(displayed_components) == 1:
            lbl, _sig, _comps = displayed_components[0]
            self._wire_label.setText(lbl)
            self._plot_widget.setTitle(f"Spectrum — {lbl}")

    def _draw_impulse(self, freq: float, power_dbm: float, color: Tuple, name: str,
                      is_carrier: bool = False, tooltip: str = "",
                      legend_name: Optional[str] = ..., x_span: float = 1e6) -> None:
        """
        Draw a vertical impulse line with enhanced styling.

        Parameters
        ----------
        legend_name : str or None or Ellipsis
            If Ellipsis (default), use *name* for the legend entry.
            If None, suppress the legend entry.
            Otherwise use the given string.
        """
        bottom = self._impulse_floor if self._impulse_floor is not None else power_dbm - 60
        line_width = 3.0 if is_carrier else 2.0

        # Determine what goes in the legend
        if legend_name is ...:
            stem_name = name
        else:
            stem_name = legend_name  # None → no legend entry

        stem = pg.PlotDataItem(
            [freq, freq], [bottom, power_dbm],
            pen=pg.mkPen(color, width=line_width),
            name=stem_name,
        )
        self._plot_widget.addItem(stem)

        cap_half_width = max(x_span * 0.0025, 1.0)
        base_cap = pg.PlotDataItem(
            [freq - cap_half_width, freq + cap_half_width], [bottom, bottom],
            pen=pg.mkPen(color, width=line_width),
        )
        self._plot_widget.addItem(base_cap)

        dot_size = 8 if is_carrier else 6
        dot = pg.ScatterPlotItem(
            x=[freq], y=[power_dbm],
            size=dot_size, brush=pg.mkBrush(color), pen=pg.mkPen(color),
            hoverable=True, hoverSize=dot_size + 3,
        )
        dot.setToolTip(f"{name}: {power_dbm:.1f} dBm @ {tooltip if tooltip else freq:.2e} Hz")
        self._plot_widget.addItem(dot)
# ======================================================================= #
# Frequency Response View - with Source/Sink Selection                    #
# ======================================================================= #

class FrequencyResponseView(QWidget):
    """
    Shows frequency response (Gain/NF) between selected source and sink blocks.
    
    Includes source and sink selection dropdowns to dynamically update the view.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frequency Response Between Source and Sink")
        self.resize(1000, 650)
        layout = QVBoxLayout(self)

        # Selection layout
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Source:"))
        self._source_combo = QComboBox()
        self._source_combo.currentIndexChanged.connect(self._on_selection_changed)
        sel_layout.addWidget(self._source_combo)
        
        sel_layout.addSpacing(20)
        sel_layout.addWidget(QLabel("Sink:"))
        self._sink_combo = QComboBox()
        self._sink_combo.currentIndexChanged.connect(self._on_selection_changed)
        sel_layout.addWidget(self._sink_combo)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)

        # Checkbox controls
        ctrl_layout = QHBoxLayout()
        self._cb_gain = QCheckBox("Gain")
        self._cb_gain.setChecked(True)
        self._cb_nf = QCheckBox("NF")
        self._cb_nf.setChecked(True)
        self._cb_p1db = QCheckBox("P1dB (input)")
        self._cb_p1db.setChecked(True)
        self._cb_oip3 = QCheckBox("OIP3")
        self._cb_oip3.setChecked(True)
        for cb in (self._cb_gain, self._cb_nf, self._cb_p1db, self._cb_oip3):
            cb.stateChanged.connect(self._refresh)
            ctrl_layout.addWidget(cb)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self._plot_widget.setLabel("left", "Level", units="dB / dBm")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.addLegend()
        layout.addWidget(self._plot_widget)

        self._freq_hz: Optional[np.ndarray] = None
        self._gain_trace: Optional[np.ndarray] = None
        self._nf_trace: Optional[np.ndarray] = None
        self._p1db_value: Optional[float] = None
        self._oip3_value: Optional[float] = None
        
        self._sources: List[Tuple[str, str]] = []  # (label, block_id)
        self._sinks: List[Tuple[str, str]] = []    # (label, block_id)
        self._on_selection_changed_callback: Optional[callable] = None

    def set_sources_and_sinks(self, sources: List[Tuple[str, str]], sinks: List[Tuple[str, str]]) -> None:
        """
        Set available sources and sinks.
        
        Parameters
        ----------
        sources : list of (label, block_id) tuples
        sinks : list of (label, block_id) tuples
        """
        self._sources = sources
        self._sinks = sinks
        
        self._source_combo.blockSignals(True)
        self._sink_combo.blockSignals(True)
        
        self._source_combo.clear()
        for label, _ in sources:
            self._source_combo.addItem(label)
        
        self._sink_combo.clear()
        for label, _ in sinks:
            self._sink_combo.addItem(label)
        
        self._source_combo.blockSignals(False)
        self._sink_combo.blockSignals(False)
        
        self._on_selection_changed()

    def set_data(
        self,
        freq_hz: np.ndarray,
        gain_db: np.ndarray,
        nf_db: Optional[np.ndarray] = None,
        p1db_dbm: Optional[float] = None,
        oip3_dbm: Optional[float] = None,
    ) -> None:
        """Update the plot data."""
        self._freq_hz = freq_hz
        self._gain_trace = gain_db
        self._nf_trace = nf_db
        self._p1db_value = p1db_dbm
        self._oip3_value = oip3_dbm
        self._refresh()

    def get_selected_source_id(self) -> Optional[str]:
        """Get the currently selected source block ID."""
        idx = self._source_combo.currentIndex()
        if idx >= 0 and idx < len(self._sources):
            return self._sources[idx][1]
        return None

    def get_selected_sink_id(self) -> Optional[str]:
        """Get the currently selected sink block ID."""
        idx = self._sink_combo.currentIndex()
        if idx >= 0 and idx < len(self._sinks):
            return self._sinks[idx][1]
        return None

    def set_on_selection_changed(self, callback: callable) -> None:
        """Set callback to be called when source or sink selection changes."""
        self._on_selection_changed_callback = callback

    def _on_selection_changed(self) -> None:
        """Handle source/sink selection changes."""
        if self._on_selection_changed_callback:
            self._on_selection_changed_callback()

    def _refresh(self) -> None:
        """Refresh the plot based on checkbox states."""
        self._plot_widget.clear()
        if self._freq_hz is None:
            return

        if self._cb_gain.isChecked() and self._gain_trace is not None:
            self._plot_widget.plot(
                self._freq_hz, self._gain_trace,
                pen=pg.mkPen("c", width=2.5), name="Gain"
            )

        if self._cb_nf.isChecked() and self._nf_trace is not None:
            self._plot_widget.plot(
                self._freq_hz, self._nf_trace,
                pen=pg.mkPen("y", width=2.5), name="NF"
            )

        if self._cb_p1db.isChecked() and self._p1db_value is not None:
            line = pg.InfiniteLine(pos=self._p1db_value, angle=0,
                                   pen=pg.mkPen("g", width=1.5, style=Qt.DashLine),
                                   label=f"P1dB={self._p1db_value:.1f} dBm")
            self._plot_widget.addItem(line)

        if self._cb_oip3.isChecked() and self._oip3_value is not None:
            line = pg.InfiniteLine(pos=self._oip3_value, angle=0,
                                   pen=pg.mkPen("m", width=1.5, style=Qt.DashLine),
                                   label=f"OIP3={self._oip3_value:.1f} dBm")
            self._plot_widget.addItem(line)


# ======================================================================= #
# Frequency Component Editor - For Mixers and Amplifiers                  #
# ======================================================================= #

class FrequencyComponentEditor(QDialog):
    """
    Dialog for editing frequency components of Mixer and Amplifier blocks.
    
    For Amplifiers: defines frequency components relative to input frequency (m*f_in + n*df)
    For Mixers: defines frequency components relative to RF and LO inputs
    """

    def __init__(self, block, parent=None):
        super().__init__(parent)
        self.block = block
        self.setWindowTitle(f"Frequency Components - {block.label}")
        self.resize(700, 500)
        layout = QVBoxLayout(self)

        # Title
        block_type = block.BLOCK_TYPE
        title = QLabel(f"{block_type}: {block.label}")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title)

        if block_type == "Amplifier":
            self._setup_amplifier_editor(layout)
        elif block_type == "Mixer":
            self._setup_mixer_editor(layout)
        else:
            layout.addWidget(QLabel("This block type doesn't support frequency component editing."))
            layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _setup_amplifier_editor(self, layout: QVBoxLayout) -> None:
        """Setup editor for Amplifier frequency components."""
        info = QLabel(
            "Amplifier frequency components are defined relative to the input frequency.\n"
            "By default, the gain is applied at m=1, n=0 (the input frequency itself).\n"
            "Add additional components as spurs with their relative power."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #AABBAA;")
        layout.addWidget(info)

        # Table for frequency components
        self._comp_table = QTableWidget()
        self._comp_table.setColumnCount(4)
        self._comp_table.setHorizontalHeaderLabels(["m", "n", "Rel Freq (m·f_in + n·Δf)", "Rel Power (dB)"])
        self._comp_table.horizontalHeader().setStretchLastSection(True)
        self._comp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self._comp_table)

        # Load existing spur coefficients
        if hasattr(self.block, 'spur_coefficients'):
            for coeff in self.block.spur_coefficients:
                self._add_component_row(
                    m=coeff.get("m", 1),
                    n=coeff.get("n", 0),
                    rel_power_db=coeff.get("rel_power_db", -60.0),
                )

        # Add default main component (m=1, n=0) if not present
        if not hasattr(self.block, 'spur_coefficients') or len(self.block.spur_coefficients) == 0:
            self._add_component_row(m=1, n=0, rel_power_db=0.0)

        # Add/Remove buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Component")
        add_btn.clicked.connect(lambda: self._add_component_row())
        btn_layout.addWidget(add_btn)

        remove_btn = QPushButton("- Remove Selected")
        remove_btn.clicked.connect(lambda: self._remove_selected_row(self._comp_table))
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _setup_mixer_editor(self, layout: QVBoxLayout) -> None:
        """Setup editor for Mixer frequency components."""
        info = QLabel(
            "Mixer frequency components combine RF input frequency and LO frequency.\n"
            "Typical combinations: RF-LO (down-conversion), RF+LO (up-conversion), 2*RF-LO, etc.\n"
            "Define each output component and its relative power (conversion loss is in the gain_db field)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #AABBAA;")
        layout.addWidget(info)

        # Expressions table
        self._expr_table = QTableWidget()
        self._expr_table.setColumnCount(3)
        self._expr_table.setHorizontalHeaderLabels(["Expression", "Description", "Rel Power (dB)"])
        self._expr_table.horizontalHeader().setStretchLastSection(True)
        self._expr_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self._expr_table)

        # Load existing conversion expressions
        if hasattr(self.block, 'conversion_expressions'):
            for expr in self.block.conversion_expressions:
                self._add_expression_row(expr)
        else:
            self._add_expression_row("RF-LO")

        # Load spur coefficients
        if hasattr(self.block, 'spur_coefficients'):
            for coeff in self.block.spur_coefficients:
                m, n = coeff.get("m", 1), coeff.get("n", 0)
                rel_power_db = coeff.get("rel_power_db", -60.0)
                expr = f"{m}*RF{'+' if n >= 0 else ''}{n}*LO" if n != 0 else f"{m}*RF"
                self._add_expression_row(expr, rel_power_db)

        # Add/Remove buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Expression")
        add_btn.clicked.connect(lambda: self._add_expression_row())
        btn_layout.addWidget(add_btn)

        remove_btn = QPushButton("- Remove Selected")
        remove_btn.clicked.connect(lambda: self._remove_selected_row(self._expr_table))
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _add_component_row(self, m: float = 1, n: float = 0, rel_power_db: float = -60) -> None:
        """Add a row to the amplitude component table."""
        row = self._comp_table.rowCount()
        self._comp_table.insertRow(row)

        m_item = QSpinBox()
        m_item.setMinimum(-10)
        m_item.setMaximum(10)
        m_item.setValue(int(m))
        self._comp_table.setCellWidget(row, 0, m_item)

        n_item = QSpinBox()
        n_item.setMinimum(-10)
        n_item.setMaximum(10)
        n_item.setValue(int(n))
        self._comp_table.setCellWidget(row, 1, n_item)

        freq_label = QLabel(f"1·f_in + 0·Δf")
        self._comp_table.setCellWidget(row, 2, freq_label)

        power_item = QDoubleSpinBox()
        power_item.setMinimum(-120)
        power_item.setMaximum(60)
        power_item.setValue(rel_power_db)
        power_item.setSuffix(" dB")
        self._comp_table.setCellWidget(row, 3, power_item)

    def _add_expression_row(self, expr: str = "RF-LO", rel_power_db: float = 0) -> None:
        """Add a row to the mixer expression table."""
        row = self._expr_table.rowCount()
        self._expr_table.insertRow(row)

        expr_item = QTableWidgetItem(expr)
        self._expr_table.setItem(row, 0, expr_item)

        desc_item = QTableWidgetItem(self._describe_expression(expr))
        desc_item.setFlags(desc_item.flags() & ~Qt.ItemIsEditable)
        self._expr_table.setItem(row, 1, desc_item)

        power_item = QDoubleSpinBox()
        power_item.setMinimum(-120)
        power_item.setMaximum(60)
        power_item.setValue(rel_power_db)
        power_item.setSuffix(" dB")
        self._expr_table.setCellWidget(row, 2, power_item)

    def _remove_selected_row(self, table: QTableWidget) -> None:
        """Remove selected row from table."""
        for index in sorted([idx.row() for idx in table.selectedIndexes()], reverse=True):
            table.removeRow(index)

    def accept(self) -> None:
        """Persist edited frequency-component settings for supported block types before closing."""
        try:
            if self.block.BLOCK_TYPE == "Amplifier":
                coeffs = []
                for row in range(self._comp_table.rowCount()):
                    m_widget = self._comp_table.cellWidget(row, 0)
                    n_widget = self._comp_table.cellWidget(row, 1)
                    p_widget = self._comp_table.cellWidget(row, 3)
                    if m_widget is None or n_widget is None or p_widget is None:
                        continue
                    coeffs.append({
                        "m": int(m_widget.value()),
                        "n": int(n_widget.value()),
                        "rel_power_db": float(p_widget.value()),
                    })
                self.block.spur_coefficients = coeffs
            elif self.block.BLOCK_TYPE == "Mixer":
                expressions: List[str] = []
                coeffs = []
                from rf_tool.blocks.components import Mixer as MixerCls
                for row in range(self._expr_table.rowCount()):
                    expr_item = self._expr_table.item(row, 0)
                    p_widget = self._expr_table.cellWidget(row, 2)
                    if expr_item is None or p_widget is None:
                        continue
                    expr = expr_item.text().strip()
                    if not expr:
                        continue
                    expressions.append(expr)
                    mn = MixerCls._expr_to_mn(expr)
                    if mn is not None:
                        coeffs.append({
                            "m": int(mn[0]),
                            "n": int(mn[1]),
                            "rel_power_db": float(p_widget.value()),
                        })
                if expressions:
                    self.block.conversion_expressions = expressions
                self.block.spur_coefficients = coeffs
        except Exception as exc:
            QMessageBox.warning(self, "Frequency Component Editor", f"Could not save component edits:\n{exc}")
        super().accept()

    @staticmethod
    def _describe_expression(expr: str) -> str:
        """Provide a description of a frequency expression."""
        expr_lower = expr.upper()
        descriptions = {
            "RF-LO": "Down-conversion (SSB)",
            "RF+LO": "Up-conversion",
            "2*RF-LO": "Double sideband",
            "RF": "Direct (no mixing)",
            "LO": "LO frequency only",
        }
        return descriptions.get(expr_lower, expr)


# ======================================================================= #
# Gain / NF Plot                                                           #
# ======================================================================= #

class GainNFPlot(QWidget):
    """
    Plots Gain and NF versus frequency for a sequence of blocks.

    Also optionally overlays P1dB, OIP3, min/max power constraints.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gain / NF vs. Frequency")
        self.resize(900, 600)
        layout = QVBoxLayout(self)

        # Checkbox controls
        ctrl_layout = QHBoxLayout()
        self._cb_gain = QCheckBox("Gain")
        self._cb_gain.setChecked(True)
        self._cb_nf = QCheckBox("NF")
        self._cb_nf.setChecked(True)
        self._cb_p1db = QCheckBox("P1dB (input)")
        self._cb_p1db.setChecked(True)
        self._cb_oip3 = QCheckBox("OIP3")
        self._cb_oip3.setChecked(True)
        self._cb_damage = QCheckBox("Damage Level")
        self._cb_damage.setChecked(False)
        for cb in (self._cb_gain, self._cb_nf, self._cb_p1db, self._cb_oip3, self._cb_damage):
            cb.stateChanged.connect(self._refresh)
            ctrl_layout.addWidget(cb)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self._plot_widget.setLabel("left", "Level", units="dB / dBm")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.addLegend()
        layout.addWidget(self._plot_widget)

        self._freq_hz: Optional[np.ndarray] = None
        self._gain_trace: Optional[np.ndarray] = None
        self._nf_trace: Optional[np.ndarray] = None
        self._p1db_value: Optional[float] = None
        self._oip3_value: Optional[float] = None
        self._damage_value: Optional[float] = None

    def set_data(
        self,
        freq_hz: np.ndarray,
        gain_db: np.ndarray,
        nf_db: Optional[np.ndarray] = None,
        p1db_dbm: Optional[float] = None,
        oip3_dbm: Optional[float] = None,
        damage_dbm: Optional[float] = None,
        title: str = "Gain / NF vs. Frequency",
    ) -> None:
        self._freq_hz = freq_hz
        self._gain_trace = gain_db
        self._nf_trace = nf_db
        self._p1db_value = p1db_dbm
        self._oip3_value = oip3_dbm
        self._damage_value = damage_dbm
        self._plot_widget.setTitle(title)
        self._refresh()

    def _refresh(self) -> None:
        self._plot_widget.clear()
        if self._freq_hz is None:
            return

        if self._cb_gain.isChecked() and self._gain_trace is not None:
            self._plot_widget.plot(
                self._freq_hz, self._gain_trace,
                pen=pg.mkPen("c", width=2), name="Gain"
            )

        if self._cb_nf.isChecked() and self._nf_trace is not None:
            self._plot_widget.plot(
                self._freq_hz, self._nf_trace,
                pen=pg.mkPen("y", width=2), name="NF"
            )

        if self._cb_p1db.isChecked() and self._p1db_value is not None:
            line = pg.InfiniteLine(pos=self._p1db_value, angle=0,
                                   pen=pg.mkPen("g", width=1, style=Qt.DashLine),
                                   label=f"P1dB={self._p1db_value:.1f} dBm")
            self._plot_widget.addItem(line)

        if self._cb_oip3.isChecked() and self._oip3_value is not None:
            line = pg.InfiniteLine(pos=self._oip3_value, angle=0,
                                   pen=pg.mkPen("m", width=1, style=Qt.DashLine),
                                   label=f"OIP3={self._oip3_value:.1f} dBm")
            self._plot_widget.addItem(line)

        if self._cb_damage.isChecked() and self._damage_value is not None:
            line = pg.InfiniteLine(pos=self._damage_value, angle=0,
                                   pen=pg.mkPen("r", width=1, style=Qt.DotLine),
                                   label=f"Damage={self._damage_value:.1f} dBm")
            self._plot_widget.addItem(line)


# ======================================================================= #
# Frequency-sweep utilities for SparBlock and TransferFnBlock             #
# ======================================================================= #

def compute_frequency_sweep(
    blocks: list,
    freq_start: float = 100e6,
    freq_stop: float = 10e9,
    n_points: int = 201,
) -> Dict[str, np.ndarray]:
    """
    Evaluate gain vs frequency for a list of blocks.

    For S-parameter and Transfer Function blocks, this queries each
    block at each frequency.  For scalar blocks it uses a flat gain.

    Parameters
    ----------
    blocks : list of RFBlock
    freq_start, freq_stop : float
        Frequency range in Hz.
    n_points : int

    Returns
    -------
    dict with keys:
        "freq_hz"    : np.ndarray shape (n_points,)
        "gain_db"    : np.ndarray shape (n_points,)   total gain
        "nf_db"      : np.ndarray shape (n_points,)   total NF (Friis)
    """
    from rf_tool.engine.cascade import db_to_linear_power, cascade_networks, s21_to_gain_db
    from rf_tool.blocks.components import SparBlock, TransferFnBlock, LowPassFilter, HighPassFilter
    try:
        import skrf  # type: ignore
    except ImportError:
        skrf = None

    freqs = np.linspace(freq_start, freq_stop, n_points)
    total_gain = np.zeros(n_points)
    total_nf_linear = np.ones(n_points)    # F_total running sum
    cum_gain_linear = np.ones(n_points)    # G1*G2*...*G(k-1)
    stage_gain_arrays: List[np.ndarray] = []
    stage_networks = []

    for block in blocks:
        # Per-frequency gain
        if isinstance(block, SparBlock) and block._network is not None:
            gain_arr = np.array([block.get_gain_db_at(f) for f in freqs])
        elif isinstance(block, TransferFnBlock):
            gain_arr = np.array([block.gain_db_at_freq(f) for f in freqs])
        elif isinstance(block, LowPassFilter):
            gain_arr = np.array([block.gain_db_at_freq(f) for f in freqs])
        elif isinstance(block, HighPassFilter):
            gain_arr = np.array([block.gain_db_at_freq(f) for f in freqs])
        else:
            gain_arr = np.full(n_points, block.gain_db)

        stage_gain_arrays.append(gain_arr)
        total_gain += gain_arr
        s21_mag = 10.0 ** (gain_arr / 20.0)
        s = np.zeros((n_points, 2, 2), dtype=complex)
        s[:, 1, 0] = s21_mag
        s[:, 0, 1] = s21_mag
        if skrf is not None:
            try:
                freq_obj = skrf.Frequency.from_f(freqs, unit="hz")
                stage_networks.append(skrf.Network(frequency=freq_obj, s=s))
            except Exception as exc:
                logger.warning("compute_frequency_sweep: disabling network cascade fallback due to stage build error: %s", exc)
                stage_networks = []
                skrf = None

        # Friis NF update: F_total += (F_block - 1) / cum_gain
        F_block = db_to_linear_power(block.nf_db)
        G_arr = 10.0 ** (gain_arr / 10.0)
        total_nf_linear += (F_block - 1.0) / cum_gain_linear
        cum_gain_linear *= G_arr

    if stage_networks:
        try:
            total_gain = s21_to_gain_db(cascade_networks(stage_networks))
        except Exception as exc:
            logger.warning("compute_frequency_sweep: falling back to additive gains (network cascade failed): %s", exc)
            total_gain = np.sum(stage_gain_arrays, axis=0) if stage_gain_arrays else np.zeros(n_points)

    total_nf_db = 10.0 * np.log10(np.maximum(total_nf_linear, 1e-300))
    return {
        "freq_hz": freqs,
        "gain_db": total_gain,
        "nf_db": total_nf_db,
    }
