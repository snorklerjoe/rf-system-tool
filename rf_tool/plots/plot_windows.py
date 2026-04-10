"""
pyqtgraph-based plot windows for RF System Tool.

SpectrumPlot   - Shows frequency-domain spectrum at a Sink node
GainNFPlot     - Shows Gain / NF vs. frequency for a P2P path
"""
from __future__ import annotations

import math
from typing import List, Optional, Dict

import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox
from PySide6.QtCore import Qt

import pyqtgraph as pg
pg.setConfigOption("background", "k")
pg.setConfigOption("foreground", "w")


# ======================================================================= #
# Spectrum Plot                                                            #
# ======================================================================= #

class SpectrumPlot(QWidget):
    """
    Shows the frequency-domain spectrum at a Sink node.

    Draws scalar impulse lines for each tone (carrier + spurs).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrum Viewer")
        self.resize(800, 400)
        layout = QVBoxLayout(self)

        self._plot_widget = pg.PlotWidget(title="Spectrum at Node")
        self._plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self._plot_widget.setLabel("left", "Power", units="dBm")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot_widget)

        self._info_label = QLabel("")
        self._info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._info_label)

    def set_signal(self, signal) -> None:
        """
        Populate the plot from a Signal object.

        Parameters
        ----------
        signal : Signal
        """
        self._plot_widget.clear()

        fc = signal.carrier_frequency
        pw = signal.power_dbm

        # Draw carrier impulse
        self._draw_impulse(fc, pw, color=(0, 200, 255), name="Carrier")

        # Draw spurs
        for spur in signal.spurs:
            self._draw_impulse(spur.frequency, spur.power_dbm, color=(255, 100, 0), name="Spur")

        self._plot_widget.setTitle(
            f"Spectrum — Carrier: {fc/1e9:.4f} GHz, {pw:.1f} dBm, "
            f"Spurs: {len(signal.spurs)}"
        )
        self._info_label.setText(
            f"Carrier: {fc/1e9:.4f} GHz  |  Power: {pw:.1f} dBm  |  "
            f"Spurs: {len(signal.spurs)}"
        )

    def _draw_impulse(self, freq: float, power_dbm: float, color, name: str) -> None:
        """Draw a vertical line from bottom of plot to power_dbm at freq."""
        bottom = self._plot_widget.viewRange()[1][0] if self._plot_widget.viewRange()[1][0] < power_dbm - 40 else power_dbm - 60
        item = pg.PlotDataItem(
            [freq, freq], [bottom, power_dbm],
            pen=pg.mkPen(color, width=2),
            name=name,
        )
        self._plot_widget.addItem(item)
        # Small horizontal tick at top
        tick = pg.PlotDataItem(
            [freq - abs(freq) * 0.005, freq + abs(freq) * 0.005],
            [power_dbm, power_dbm],
            pen=pg.mkPen(color, width=2),
        )
        self._plot_widget.addItem(tick)


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
        self._cb_gain = QCheckBox("Gain", checked=True)
        self._cb_nf = QCheckBox("NF", checked=True)
        self._cb_p1db = QCheckBox("P1dB (input)", checked=False)
        self._cb_oip3 = QCheckBox("OIP3", checked=False)
        self._cb_damage = QCheckBox("Damage Level", checked=False)
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
    from rf_tool.engine.cascade import (
        db_to_linear_power, linear_power_to_db,
    )
    from rf_tool.blocks.components import SparBlock, TransferFnBlock, LowPassFilter, HighPassFilter

    freqs = np.linspace(freq_start, freq_stop, n_points)
    total_gain = np.zeros(n_points)
    total_nf_linear = np.ones(n_points)    # F_total running sum
    cum_gain_linear = np.ones(n_points)    # G1*G2*...*G(k-1)

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

        total_gain += gain_arr

        # Friis NF update: F_total += (F_block - 1) / cum_gain
        F_block = db_to_linear_power(block.nf_db)
        G_arr = 10.0 ** (gain_arr / 10.0)
        total_nf_linear += (F_block - 1.0) / cum_gain_linear
        cum_gain_linear *= G_arr

    total_nf_db = 10.0 * np.log10(np.maximum(total_nf_linear, 1e-300))
    return {
        "freq_hz": freqs,
        "gain_db": total_gain,
        "nf_db": total_nf_db,
    }
