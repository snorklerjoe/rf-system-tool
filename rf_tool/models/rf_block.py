"""
RFBlock base class for all RF components.

Every component:
  - Has X/Y position, color, and label.
  - Exposes input and output port lists.
  - Carries small-signal parameters (gain, NF) and large-signal
    parameters (P1dB, OIP3).
  - Can process a Signal and return a modified Signal.
  - Defines min/max input power damage thresholds.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class Port:
    """Represents a single input or output port on an RFBlock."""
    name: str
    direction: str      # "input" or "output"
    port_index: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "direction": self.direction, "port_index": self.port_index}

    @classmethod
    def from_dict(cls, d: dict) -> "Port":
        return cls(name=d["name"], direction=d["direction"], port_index=d.get("port_index", 0))


class RFBlock:
    """
    Base class for all RF system components.

    Parameters
    ----------
    block_id : str, optional
        Unique identifier (auto-generated UUID if not provided).
    label : str
        Human-readable label shown on the canvas.
    x, y : float
        Canvas position.
    color : str
        Block colour in CSS hex format (e.g. "#4080FF").
    gain_db : float
        Small-signal gain (positive = amplification, negative = loss).
    nf_db : float
        Noise figure in dB (0 = noiseless).
    p1db_dbm : float or None
        Output 1-dB compression point (dBm).  None = linear.
    oip3_dbm : float or None
        Output third-order intercept point (dBm).  None = linear.
    min_input_power_dbm : float or None
        Minimum safe input power (below this = warning).
    max_input_power_dbm : float or None
        Maximum safe input power (above this = damage).
    spur_coefficients : list of dict
        Each entry: {"m": int, "n": int, "rel_power_db": float}
        meaning output spur at m*f_rf + n*f_lo has relative power
        rel_power_db dB below the output carrier.
    """

    BLOCK_TYPE: str = "RFBlock"     # subclasses override this

    def __init__(
        self,
        block_id: Optional[str] = None,
        label: str = "Block",
        x: float = 0.0,
        y: float = 0.0,
        color: str = "#4080FF",
        gain_db: float = 0.0,
        nf_db: float = 0.0,
        p1db_dbm: Optional[float] = None,
        oip3_dbm: Optional[float] = None,
        min_input_power_dbm: Optional[float] = None,
        max_input_power_dbm: Optional[float] = None,
        spur_coefficients: Optional[List[Dict]] = None,
        comment_mode: str = "active",
    ) -> None:
        self.block_id: str = block_id or str(uuid.uuid4())
        self.label: str = label
        self.x: float = x
        self.y: float = y
        self.color: str = color

        # Small-signal
        self.gain_db: float = gain_db
        self.nf_db: float = nf_db

        # Large-signal
        self.p1db_dbm: Optional[float] = p1db_dbm
        self.oip3_dbm: Optional[float] = oip3_dbm

        # Damage thresholds
        self.min_input_power_dbm: Optional[float] = min_input_power_dbm
        self.max_input_power_dbm: Optional[float] = max_input_power_dbm

        # Spurious generation
        self.spur_coefficients: List[Dict] = spur_coefficients or []
        self.comment_mode: str = comment_mode if comment_mode in {"active", "out", "through"} else "active"

        # Ports are defined by subclasses
        self._input_ports: List[Port] = []
        self._output_ports: List[Port] = []
        self._setup_ports()

    # ------------------------------------------------------------------ #
    # Port management                                                      #
    # ------------------------------------------------------------------ #
    def _setup_ports(self) -> None:
        """Subclasses define their ports here."""
        self._input_ports = [Port("IN", "input", 0)]
        self._output_ports = [Port("OUT", "output", 0)]

    @property
    def input_ports(self) -> List[Port]:
        return self._input_ports

    @property
    def output_ports(self) -> List[Port]:
        return self._output_ports

    # ------------------------------------------------------------------ #
    # Signal processing                                                    #
    # ------------------------------------------------------------------ #
    def process(self, signal, port_name: str = "IN"):
        """
        Process an incoming signal and return the output signal(s).

        Returns
        -------
        dict mapping output port name -> Signal, or a single Signal.
        """
        from rf_tool.models.signal import Signal
        out = signal.apply_gain(self.gain_db)
        # Add spurs if defined
        for coeff in self.spur_coefficients:
            m = coeff.get("m", 1)
            n = coeff.get("n", 0)
            rel_db = coeff.get("rel_power_db", -60.0)
            f_spur = m * signal.carrier_frequency
            p_spur = signal.power_dbm + self.gain_db + rel_db
            out.add_spur(f_spur, p_spur)
        return {"OUT": out}

    # ------------------------------------------------------------------ #
    # Power validation                                                     #
    # ------------------------------------------------------------------ #
    def check_power(self, input_power_dbm: float) -> str:
        """
        Check if input power is within safe operating range.

        Returns
        -------
        "ok", "low", or "high"
        """
        if self.max_input_power_dbm is not None and input_power_dbm > self.max_input_power_dbm:
            return "high"
        if self.min_input_power_dbm is not None and input_power_dbm < self.min_input_power_dbm:
            return "low"
        return "ok"

    # ------------------------------------------------------------------ #
    # Derived quantities                                                   #
    # ------------------------------------------------------------------ #
    @property
    def gain_linear(self) -> float:
        """Linear power gain."""
        import math
        return 10 ** (self.gain_db / 10.0)

    @property
    def nf_linear(self) -> float:
        """Linear noise factor (F = 10^(NF_dB/10))."""
        import math
        return 10 ** (self.nf_db / 10.0)

    @property
    def oip3_dbm_inferred(self) -> Optional[float]:
        """Return OIP3 or estimate from P1dB (OIP3 ≈ P1dB + 9.6 dB)."""
        if self.oip3_dbm is not None:
            return self.oip3_dbm
        if self.p1db_dbm is not None:
            return self.p1db_dbm + 9.6
        return None

    @property
    def iip3_dbm(self) -> Optional[float]:
        """Input-referred IP3 = OIP3 - gain."""
        oip3 = self.oip3_dbm_inferred
        if oip3 is not None:
            return oip3 - self.gain_db
        return None

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "block_type": self.BLOCK_TYPE,
            "block_id": self.block_id,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "gain_db": self.gain_db,
            "nf_db": self.nf_db,
            "p1db_dbm": self.p1db_dbm,
            "oip3_dbm": self.oip3_dbm,
            "min_input_power_dbm": self.min_input_power_dbm,
            "max_input_power_dbm": self.max_input_power_dbm,
            "spur_coefficients": self.spur_coefficients,
            "comment_mode": self.comment_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RFBlock":
        return cls(
            block_id=d.get("block_id"),
            label=d.get("label", "Block"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#4080FF"),
            gain_db=d.get("gain_db", 0.0),
            nf_db=d.get("nf_db", 0.0),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.block_id[:8]}, "
            f"label={self.label!r}, gain={self.gain_db} dB)"
        )
