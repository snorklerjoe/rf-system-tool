"""
Hierarchical block types for multi-level schematic design.

HierInputPin  – labelled input pin on a hierarchical sheet (one output port)
HierOutputPin – labelled output pin on a hierarchical sheet (one input port)
HierSubcircuit – reference to an external JSON scene file; ports are loaded
                 dynamically from the referenced file's pin blocks.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Any, Tuple

from rf_tool.models.rf_block import RFBlock, Port
from rf_tool.models.signal import Signal

_MIN_POWER_MW = 1e-30
_MIN_PROPAGATION_ITERATIONS = 50
_ITERATIONS_PER_CONNECTION = 8


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


def _signals_equivalent(a: Optional[Signal], b: Optional[Signal]) -> bool:
    """Compare two Signal instances with practical tolerances."""
    if a is b:
        return True
    if a is None or b is None:
        return False
    if (
        abs(a.carrier_frequency - b.carrier_frequency) > 1e-6
        or abs(a.power_dbm - b.power_dbm) > 1e-6
    ):
        return False
    if len(a.spurs) != len(b.spurs):
        return False
    a_spurs = sorted(a.spurs, key=lambda s: (s.frequency, s.power_dbm))
    b_spurs = sorted(b.spurs, key=lambda s: (s.frequency, s.power_dbm))
    for sa, sb in zip(a_spurs, b_spurs):
        if abs(sa.frequency - sb.frequency) > 1e-6 or abs(sa.power_dbm - sb.power_dbm) > 1e-6:
            return False
    return True


def _merge_signals(existing: Optional[Signal], incoming: Signal) -> Signal:
    """Power-combine two signals at the same node, including spurs/noise."""
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
        [(f_hz, 10.0 * math.log10(max(p_mw, _MIN_POWER_MW))) for f_hz, p_mw in tone_bins],
        key=lambda x: x[1],
        reverse=True,
    )
    carrier_f, carrier_p = combined_tones[0]
    out = Signal(carrier_frequency=carrier_f, power_dbm=carrier_p, spurs=[])
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
            out.snr_db = 10.0 * math.log10(total_signal / max(total_noise, _MIN_POWER_MW))
        return out

    noise_terms = []
    if existing_nf is not None:
        noise_terms.append(10.0 ** (existing_nf / 10.0))
    if incoming_nf is not None:
        noise_terms.append(10.0 ** (incoming_nf / 10.0))
    total_noise_mw = sum(noise_terms)
    out_noise_floor = 10.0 * math.log10(max(total_noise_mw, _MIN_POWER_MW))
    out.set_noise_floor_dbm(out_noise_floor)
    return out


def _resolve_subcircuit_path(path: str, base_dir: str) -> str:
    """Resolve a potentially relative subcircuit file path against base_dir."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


class HierSubcircuit(RFBlock):
    """
    A hierarchical sub-circuit block that references an external JSON scene file.

    Ports are loaded dynamically from the referenced file's HierInputPin /
    HierOutputPin blocks.  If the file is missing, the block is shown with
    a "MISSING" visual state and no functional ports.
    """

    BLOCK_TYPE = "HierSubcircuit"
    _ACTIVE_PATH_STACK: List[str] = []

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
        self._external_inputs: Dict[str, Signal] = {}

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
        self._external_inputs[port_name] = signal.copy()
        if self.file_missing or not self.subcircuit_path:
            return {}

        resolved_path = os.path.abspath(self.subcircuit_path)
        if resolved_path in self._ACTIVE_PATH_STACK:
            # Prevent recursive self-reference loops.
            return {}

        self._ACTIVE_PATH_STACK.append(resolved_path)
        try:
            return self._simulate_subcircuit()
        except Exception:
            return {}
        finally:
            self._ACTIVE_PATH_STACK.pop()

    def _simulate_subcircuit(self) -> Dict[str, Signal]:
        """Run event-based propagation inside the referenced subcircuit scene."""
        from rf_tool.blocks.components import block_from_dict

        with open(self.subcircuit_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        base_dir = os.path.dirname(os.path.abspath(self.subcircuit_path))

        blocks: Dict[str, RFBlock] = {}
        input_blocks: Dict[str, HierInputPin] = {}
        output_blocks: Dict[str, HierOutputPin] = {}

        for block_dict in data.get("blocks", []):
            block = block_from_dict(block_dict)
            if isinstance(block, HierSubcircuit):
                block.subcircuit_path = _resolve_subcircuit_path(block.subcircuit_path, base_dir)
                block.reload()
            blocks[block.block_id] = block
            if isinstance(block, HierInputPin):
                input_blocks[block.pin_name] = block
            elif isinstance(block, HierOutputPin):
                block.last_signal = None
                output_blocks[block.pin_name] = block

        connections = data.get("connections", [])
        adjacency: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for conn in connections:
            src_key = (conn["src_block_id"], conn["src_port"])
            adjacency.setdefault(src_key, []).append((conn["dst_block_id"], conn["dst_port"]))

        signals_at: Dict[str, Dict[str, Signal]] = {}
        queue: List[Tuple[str, str, Signal]] = []

        for pin_name, pin_block in input_blocks.items():
            incoming = self._external_inputs.get(pin_name)
            if incoming is None or pin_block.comment_mode == "out":
                continue
            produced = pin_block.process(incoming, pin_name)
            for out_port, out_sig in produced.items():
                signals_at.setdefault(pin_block.block_id, {})[out_port] = out_sig
                queue.append((pin_block.block_id, out_port, out_sig))

        max_iterations = max(
            _MIN_PROPAGATION_ITERATIONS,
            len(connections) * _ITERATIONS_PER_CONNECTION,
        )
        iterations = 0
        while queue:
            iterations += 1
            if iterations > max_iterations:
                break
            src_bid, src_port, sig = queue.pop(0)
            for dst_bid, dst_port in adjacency.get((src_bid, src_port), []):
                dst_block = blocks.get(dst_bid)
                if dst_block is None:
                    continue
                merged_in = _merge_signals(signals_at.setdefault(dst_bid, {}).get(dst_port), sig)
                prev_in = signals_at.setdefault(dst_bid, {}).get(dst_port)
                signals_at.setdefault(dst_bid, {})[dst_port] = merged_in
                if _signals_equivalent(prev_in, merged_in):
                    continue
                if dst_block.comment_mode == "out":
                    continue
                if dst_block.comment_mode == "through":
                    result = {p.name: merged_in.copy() for p in dst_block.output_ports}
                else:
                    result = dst_block.process(merged_in, dst_port)
                    for out_sig in result.values():
                        in_noise_floor = merged_in.get_noise_floor_dbm()
                        if in_noise_floor is not None:
                            effective_gain = out_sig.power_dbm - merged_in.power_dbm
                            out_noise_floor = in_noise_floor + effective_gain + max(0.0, dst_block.nf_db)
                            out_sig.set_noise_floor_dbm(out_noise_floor)
                        elif merged_in.snr_db is not None and out_sig.snr_db is None:
                            out_sig.snr_db = merged_in.snr_db - max(0.0, dst_block.nf_db)
                for out_port, out_sig in result.items():
                    prev_out = signals_at.setdefault(dst_bid, {}).get(out_port)
                    if _signals_equivalent(prev_out, out_sig):
                        continue
                    signals_at.setdefault(dst_bid, {})[out_port] = out_sig
                    queue.append((dst_bid, out_port, out_sig))

        outputs: Dict[str, Signal] = {}
        for pin_name, out_block in output_blocks.items():
            if out_block.last_signal is not None:
                outputs[pin_name] = out_block.last_signal.copy()
        return outputs

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
