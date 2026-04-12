"""
Ribbon-style toolbar widget for RF System Tool.

RibbonWidget embeds a QTabWidget styled as an Office-like ribbon.  An
always-visible row of utility buttons sits above the tabs via a corner
widget arrangement so that they remain accessible regardless of which
tab is active.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PySide6.QtWidgets import (
    QWidget, QTabWidget, QToolButton, QHBoxLayout, QVBoxLayout,
    QPushButton, QSizePolicy, QFrame, QTabBar,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPainter, QColor, QPen

# ─── Style constants ────────────────────────────────────────────────────────

_ALWAYS_VISIBLE_QSS = """
    QPushButton {
        background: #2E4070;
        color: #FFFFFF;
        border: 1px solid #4A6090;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    QPushButton:hover { background: #3A5490; }
    QPushButton:pressed { background: #1E3060; }
"""

_TAB_QSS = """
    QTabWidget::pane { border: none; margin: 0px; padding: 0px; }
"""

_BTN_BASE = """
    QPushButton {{
        background: {bg};
        color: #FFFFFF;
        border: 1px solid {border};
        padding: 5px 8px;
        border-radius: 3px;
        min-width: 44px;
        font-size: 12px;
    }}
    QPushButton:hover {{ background: {hover}; }}
    QPushButton:pressed {{ background: {pressed}; }}
"""

_BLUE   = _BTN_BASE.format(bg="#2C4E8A", border="#3A6AB5", hover="#3A6AB5", pressed="#1E3870")
_GREEN  = _BTN_BASE.format(bg="#1E6B3C", border="#2A8A4E", hover="#2A8A4E", pressed="#145230")
_ORANGE = _BTN_BASE.format(bg="#7A4010", border="#AA5820", hover="#AA5820", pressed="#5A2C08")
_PURPLE = _BTN_BASE.format(bg="#5A2A7A", border="#7A4AAA", hover="#7A4AAA", pressed="#3A1A5A")
_TAB_COLORS = {
    0: ("#2C4E8A", "#3A6AB5"),
    1: ("#1E6B3C", "#2A8A4E"),
    2: ("#7A4010", "#AA5820"),
    3: ("#5A2A7A", "#7A4AAA"),
}


class _ColoredTabBar(QTabBar):
    """Color each tab to match its ribbon section."""

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        for index in range(self.count()):
            rect = self.tabRect(index)
            bg, border = _TAB_COLORS.get(index, ("#2D2D44", "#44445A"))
            if index != self.currentIndex():
                bg = "#242433"
            painter.fillRect(rect, QColor(bg))
            painter.setPen(QPen(QColor(border), 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            painter.setPen(QPen(Qt.white, 1))
            painter.drawText(rect, Qt.AlignCenter, self.tabText(index))

# ─── Helper ──────────────────────────────────────────────────────────────────

def _make_row(*buttons) -> QWidget:
    """Return a QWidget with the given buttons laid out horizontally."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(4, 2, 4, 2)
    lay.setSpacing(4)
    for btn in buttons:
        if btn is None:
            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            sep.setStyleSheet("QFrame { color: #555; }")
            lay.addWidget(sep)
        else:
            lay.addWidget(btn)
    lay.addStretch(1)
    return w


def _btn(label: str, tooltip: str, qss: str, callback: Optional[Callable] = None) -> QPushButton:
    b = QPushButton(label)
    b.setToolTip(tooltip)
    b.setStyleSheet(qss)
    b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    if callback:
        b.clicked.connect(callback)
    return b


# ======================================================================= #
# RibbonWidget                                                             #
# ======================================================================= #

class RibbonWidget(QWidget):
    """
    Ribbon-style compound widget.

    Exposes a public Signal for each major action so that the main window
    can connect them without reaching into the ribbon internals.
    """

    # Always-visible
    sig_zoom_fit   = Signal()
    sig_propagate  = Signal()

    # Components tab
    sig_add_source      = Signal()
    sig_add_sink        = Signal()
    sig_add_amplifier   = Signal()
    sig_add_attenuator  = Signal()
    sig_add_mixer       = Signal()
    sig_add_switch      = Signal()
    sig_add_spar        = Signal()
    sig_add_transfer_fn = Signal()
    sig_add_lpf         = Signal()
    sig_add_hpf         = Signal()
    sig_add_splitter    = Signal()
    sig_add_combiner    = Signal()
    sig_add_annotation  = Signal()

    # Analysis tab
    sig_p2p_cascade     = Signal()
    sig_freq_plot       = Signal()
    sig_signal_spectrum = Signal()
    sig_freq_response   = Signal()

    # Tools tab
    sig_cut             = Signal()
    sig_copy            = Signal()
    sig_paste           = Signal()
    sig_comment_out     = Signal()
    sig_comment_through = Signal()
    sig_uncomment       = Signal()
    sig_select_all      = Signal()
    sig_delete_selected = Signal()

    # Hierarchical tab
    sig_add_hier_input  = Signal()
    sig_add_hier_output = Signal()
    sig_open_symbol_editor = Signal()
    sig_reload_all      = Signal()
    # subcircuit buttons emit (path,) via a generic signal
    sig_add_subcircuit  = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._subcircuit_btns: List[QPushButton] = []
        self._build_ui()

    # ------------------------------------------------------------------ #
    # Build                                                                #
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Always-visible row ───────────────────────────────────────────
        always_row = QWidget()
        always_row.setStyleSheet("QWidget { background: #1A1A30; }")
        always_lay = QHBoxLayout(always_row)
        always_lay.setContentsMargins(4, 2, 4, 2)
        always_lay.setSpacing(6)

        btn_fit = QPushButton("Zoom Fit")
        btn_fit.setToolTip("Zoom to Fit All (Home)")
        btn_fit.setStyleSheet(_ALWAYS_VISIBLE_QSS)
        btn_fit.clicked.connect(self.sig_zoom_fit)

        btn_prop = QPushButton("Propagate")
        btn_prop.setToolTip("Propagate Signals through the circuit")
        btn_prop.setStyleSheet(_ALWAYS_VISIBLE_QSS)
        btn_prop.clicked.connect(self.sig_propagate)

        always_lay.addWidget(btn_fit)
        always_lay.addWidget(btn_prop)
        always_lay.addStretch(1)
        outer.addWidget(always_row)

        # ── Tab widget ───────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setTabBar(_ColoredTabBar())
        self._tabs.setStyleSheet(_TAB_QSS)
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._tabs.addTab(self._build_components_tab(), "Components")
        self._tabs.addTab(self._build_analysis_tab(),   "Analysis")
        self._tabs.addTab(self._build_tools_tab(),      "Tools")
        self._tabs.addTab(self._build_hier_tab(),       "Hierarchical")

        outer.addWidget(self._tabs)

    # ─── Components ──────────────────────────────────────────────────────
    def _build_components_tab(self) -> QWidget:
        btns = [
            _btn("Src",      "Add Source",      _BLUE, self.sig_add_source),
            _btn("Snk",      "Add Sink",        _BLUE, self.sig_add_sink),
            None,
            _btn("Amp",      "Add Amplifier",   _BLUE, self.sig_add_amplifier),
            _btn("Att",      "Add Attenuator",  _BLUE, self.sig_add_attenuator),
            _btn("Mix",      "Add Mixer",       _BLUE, self.sig_add_mixer),
            _btn("SW",       "Add Switch",      _BLUE, self.sig_add_switch),
            None,
            _btn("[S] Spar", "Add S-Parameter Block", _BLUE, self.sig_add_spar),
            _btn("TF",       "Add Transfer Function", _BLUE, self.sig_add_transfer_fn),
            _btn("LPF",      "Add Low-Pass Filter",   _BLUE, self.sig_add_lpf),
            _btn("HPF",      "Add High-Pass Filter",  _BLUE, self.sig_add_hpf),
            None,
            _btn("Spl",      "Add Splitter",    _BLUE, self.sig_add_splitter),
            _btn("Cmb",      "Add Combiner",    _BLUE, self.sig_add_combiner),
            None,
            _btn("T Ann",    "Add Annotation",  _BLUE, self.sig_add_annotation),
        ]
        return _make_row(*btns)

    # ─── Analysis ────────────────────────────────────────────────────────
    def _build_analysis_tab(self) -> QWidget:
        btns = [
            _btn("P2P Cascade",       "Point-to-Point Cascade Analysis", _GREEN, self.sig_p2p_cascade),
            _btn("Freq Plot",         "Gain/NF vs Frequency Plot",       _GREEN, self.sig_freq_plot),
            _btn("Signal Spectrum",   "Signal Spectrum Viewer",          _GREEN, self.sig_signal_spectrum),
            _btn("Freq Response",     "Frequency Response View",         _GREEN, self.sig_freq_response),
        ]
        return _make_row(*btns)

    # ─── Tools ───────────────────────────────────────────────────────────
    def _build_tools_tab(self) -> QWidget:
        btns = [
            _btn("Cut",              "Cut selected",          _ORANGE, self.sig_cut),
            _btn("Copy",             "Copy selected",         _ORANGE, self.sig_copy),
            _btn("Paste",            "Paste clipboard",       _ORANGE, self.sig_paste),
            None,
            _btn("Comment Out",      "Comment out selected",  _ORANGE, self.sig_comment_out),
            _btn("Comment Thru",     "Comment through",       _ORANGE, self.sig_comment_through),
            _btn("Uncomment",        "Uncomment selected",    _ORANGE, self.sig_uncomment),
            None,
            _btn("Select All",       "Select all items",      _ORANGE, self.sig_select_all),
            _btn("Delete",           "Delete selected",       _ORANGE, self.sig_delete_selected),
        ]
        return _make_row(*btns)

    # ─── Hierarchical ────────────────────────────────────────────────────
    def _build_hier_tab(self) -> QWidget:
        self._hier_tab_widget = QWidget()
        self._hier_lay = QHBoxLayout(self._hier_tab_widget)
        self._hier_lay.setContentsMargins(4, 2, 4, 2)
        self._hier_lay.setSpacing(4)

        fixed_btns = [
            _btn("In Pin",      "Add Hierarchical Input Pin",   _PURPLE, self.sig_add_hier_input),
            _btn("Out Pin",     "Add Hierarchical Output Pin",  _PURPLE, self.sig_add_hier_output),
            None,
            _btn("Symbol Ed.",  "Open Symbol Editor",           _PURPLE, self.sig_open_symbol_editor),
            None,
            _btn("Reload All",  "Reload all subcircuit blocks", _PURPLE, self.sig_reload_all),
            None,
        ]
        for item in fixed_btns:
            if item is None:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet("QFrame { color: #555; }")
                self._hier_lay.addWidget(sep)
            else:
                self._hier_lay.addWidget(item)

        self._hier_lay.addStretch(1)
        return self._hier_tab_widget

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def refresh_subcircuit_buttons(self, subcircuit_list: List[Tuple[str, str]]) -> None:
        """
        Replace dynamic subcircuit buttons in the Hierarchical tab.

        Parameters
        ----------
        subcircuit_list : list of (label, path)
        """
        # Remove old dynamic buttons
        for btn in self._subcircuit_btns:
            self._hier_lay.removeWidget(btn)
            btn.deleteLater()
        self._subcircuit_btns.clear()

        # Remove trailing stretch temporarily
        stretch_item = self._hier_lay.itemAt(self._hier_lay.count() - 1)
        if stretch_item and stretch_item.spacerItem():
            self._hier_lay.removeItem(stretch_item)

        for label, path in subcircuit_list:
            btn = _btn(f"Sub: {label}", f"Add subcircuit: {path}", _PURPLE)
            btn.clicked.connect(lambda checked=False, p=path: self.sig_add_subcircuit.emit(p))
            self._hier_lay.addWidget(btn)
            self._subcircuit_btns.append(btn)

        self._hier_lay.addStretch(1)

    def set_add_component_callback(self, name: str, callback: Callable) -> None:
        """
        Dynamically connect a callback to a named component action.
        *name* should match the signal attribute, e.g. ``"sig_add_amplifier"``.
        """
        sig = getattr(self, name, None)
        if sig is not None:
            sig.connect(callback)

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width(), 120)
