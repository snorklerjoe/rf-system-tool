"""
Hierarchical block types for multi-level schematic design.

HierInputPin  – labelled input pin on a hierarchical sheet (one output port)
HierOutputPin – labelled output pin on a hierarchical sheet (one input port)
HierSubcircuit – reference to an external JSON scene file; ports are loaded
                 dynamically from the referenced file's pin blocks.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Any

from rf_tool.models.rf_block import RFBlock, Port
from rf_tool.models.signal import Signal


# ======================================================================= #
# HierInputPin                                                             #
# ======================================================================= #

class HierInputPin(RFBlock):
    """
    Input pin for a hierarchical sheet.

    Acts as a source within the sub-sheet: it provides the incoming signal
    to the rest of the circuit.  Externally it exposes a single input
    connection point named after *pin_name*.
    """

    BLOCK_TYPE = "HierInputPin"

    def __init__(self, pin_name: str = "IN", **kwargs):
        self.pin_name: str = pin_name
        kwargs.setdefault("label", pin_name)
        kwargs.setdefault("color", "#2ECC71")
        kwargs["gain_db"] = 0.0
        kwargs["nf_db"] = 0.0
        super().__init__(**kwargs)

    def _setup_ports(self) -> None:
        self._input_ports = []
        self._output_ports = [Port(self.pin_name, "output", 0)]

    def process(self, signal: Signal, port_name: str = "") -> Dict[str, Signal]:
        return {self.pin_name: signal.copy()}

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["pin_name"] = self.pin_name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HierInputPin":
        pin_name = d.get("pin_name", "IN")
        return cls(
            pin_name=pin_name,
            block_id=d.get("block_id"),
            label=d.get("label", pin_name),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#2ECC71"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            comment_mode=d.get("comment_mode", "active"),
        )


# ======================================================================= #
# HierOutputPin                                                            #
# ======================================================================= #

class HierOutputPin(RFBlock):
    """
    Output pin for a hierarchical sheet.

    Receives the signal from the circuit and exposes it as an external
    connection point named after *pin_name*.
    """

    BLOCK_TYPE = "HierOutputPin"

    def __init__(self, pin_name: str = "OUT", **kwargs):
        self.pin_name: str = pin_name
        kwargs.setdefault("label", pin_name)
        kwargs.setdefault("color", "#E74C3C")
        kwargs["gain_db"] = 0.0
        kwargs["nf_db"] = 0.0
        self.last_signal: Optional[Signal] = None
        super().__init__(**kwargs)

    def _setup_ports(self) -> None:
        self._input_ports = [Port(self.pin_name, "input", 0)]
        self._output_ports = []

    def process(self, signal: Signal, port_name: str = "") -> Dict[str, Signal]:
        self.last_signal = signal.copy()
        return {}

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["pin_name"] = self.pin_name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HierOutputPin":
        pin_name = d.get("pin_name", "OUT")
        return cls(
            pin_name=pin_name,
            block_id=d.get("block_id"),
            label=d.get("label", pin_name),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#E74C3C"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            comment_mode=d.get("comment_mode", "active"),
        )


# ======================================================================= #
# HierSubcircuit                                                           #
# ======================================================================= #

def _load_pins_from_file(path: str):
    """
    Parse *path* (a JSON scene file) and return lists of input/output pin names.

    Returns (input_pin_names, output_pin_names, file_missing, symbol).
    """
    if not os.path.isfile(path):
        return [], [], True, {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        input_pins: List[str] = []
        output_pins: List[str] = []
        for block_dict in data.get("blocks", []):
            bt = block_dict.get("block_type", "")
            pin_name = block_dict.get("pin_name") or block_dict.get("label") or "pin"
            if bt == "HierInputPin":
                input_pins.append(pin_name)
            elif bt == "HierOutputPin":
                output_pins.append(pin_name)
        metadata = data.get("metadata", {})
        symbol = metadata.get("symbol", {}) if isinstance(metadata, dict) else {}
        return input_pins, output_pins, False, symbol
    except Exception:
        return [], [], True, {}


class HierSubcircuit(RFBlock):
    """
    A hierarchical sub-circuit block that references an external JSON scene file.

    Ports are loaded dynamically from the referenced file's HierInputPin /
    HierOutputPin blocks.  If the file is missing, the block is shown with
    a "MISSING" visual state and no functional ports.
    """

    BLOCK_TYPE = "HierSubcircuit"

    def __init__(
        self,
        subcircuit_path: str = "",
        symbol: Optional[Dict] = None,
        **kwargs,
    ):
        self.subcircuit_path: str = subcircuit_path
        self.symbol: Dict = symbol or {}
        self.file_missing: bool = False
        self._input_pin_names: List[str] = []
        self._output_pin_names: List[str] = []

        kwargs.setdefault("label", os.path.splitext(os.path.basename(subcircuit_path))[0] or "Sub")
        kwargs.setdefault("color", "#8E44AD")
        kwargs["gain_db"] = 0.0
        kwargs["nf_db"] = 0.0
        super().__init__(**kwargs)

    def _setup_ports(self) -> None:
        inp, out, missing, symbol = _load_pins_from_file(self.subcircuit_path)
        self.file_missing = missing
        self._input_pin_names = inp
        self._output_pin_names = out
        if not missing:
            self.symbol = symbol or {}
        self._input_ports = [Port(n, "input", i) for i, n in enumerate(inp)]
        self._output_ports = [Port(n, "output", i) for i, n in enumerate(out)]

    def reload(self) -> None:
        """Reload port definitions from the referenced file."""
        self._setup_ports()

    def process(self, signal: Signal, port_name: str = "IN") -> Dict[str, Signal]:
        # Pass-through: a real hierarchical simulation would recurse
        result = {}
        for port in self._output_ports:
            result[port.name] = signal.copy()
        return result

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["subcircuit_path"] = self.subcircuit_path
        d["symbol"] = self.symbol
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HierSubcircuit":
        return cls(
            subcircuit_path=d.get("subcircuit_path", ""),
            symbol=d.get("symbol", {}),
            block_id=d.get("block_id"),
            label=d.get("label", "Sub"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#8E44AD"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            comment_mode=d.get("comment_mode", "active"),
        )
