"""
Signal data model for RF System Tool.

A Signal carries carrier frequency, power level, spurious tones,
and phase noise parameters through the RF signal chain.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SpurTone:
    """A single spurious tone with frequency and power level."""
    frequency: float   # Hz
    power_dbm: float   # dBm

    def scale_power(self, gain_db: float) -> "SpurTone":
        return SpurTone(frequency=self.frequency, power_dbm=self.power_dbm + gain_db)

    def to_dict(self) -> dict:
        return {"frequency": self.frequency, "power_dbm": self.power_dbm}

    @classmethod
    def from_dict(cls, d: dict) -> "SpurTone":
        return cls(frequency=d["frequency"], power_dbm=d["power_dbm"])


@dataclass
class Signal:
    """
    Represents an RF signal propagating through the system.

    Attributes
    ----------
    carrier_frequency : float
        Carrier frequency in Hz.
    power_dbm : float
        Carrier power level in dBm.
    spurs : list of SpurTone
        Spurious tones (frequency and power).
    phase_noise_dbc_hz : dict, optional
        Phase noise profile: offset_freq_hz -> dBc/Hz.
    """
    carrier_frequency: float          # Hz
    power_dbm: float                  # dBm
    spurs: List[SpurTone] = field(default_factory=list)
    phase_noise_dbc_hz: Dict[float, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Factory / copy helpers                                               #
    # ------------------------------------------------------------------ #
    def copy(self) -> "Signal":
        """Return a deep copy of this signal."""
        return Signal(
            carrier_frequency=self.carrier_frequency,
            power_dbm=self.power_dbm,
            spurs=[SpurTone(s.frequency, s.power_dbm) for s in self.spurs],
            phase_noise_dbc_hz=dict(self.phase_noise_dbc_hz),
        )

    # ------------------------------------------------------------------ #
    # Propagation helpers                                                  #
    # ------------------------------------------------------------------ #
    def apply_gain(self, gain_db: float) -> "Signal":
        """Return a new Signal with gain applied to carrier and all spurs."""
        new_sig = self.copy()
        new_sig.power_dbm += gain_db
        new_sig.spurs = [s.scale_power(gain_db) for s in new_sig.spurs]
        return new_sig

    def add_spur(self, frequency: float, power_dbm: float) -> None:
        """Add or update a spurious tone."""
        for s in self.spurs:
            if abs(s.frequency - frequency) < 1e-3:
                s.power_dbm = power_dbm
                return
        self.spurs.append(SpurTone(frequency=frequency, power_dbm=power_dbm))

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "carrier_frequency": self.carrier_frequency,
            "power_dbm": self.power_dbm,
            "spurs": [s.to_dict() for s in self.spurs],
            "phase_noise_dbc_hz": {str(k): v for k, v in self.phase_noise_dbc_hz.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Signal":
        return cls(
            carrier_frequency=d["carrier_frequency"],
            power_dbm=d["power_dbm"],
            spurs=[SpurTone.from_dict(s) for s in d.get("spurs", [])],
            phase_noise_dbc_hz={float(k): v for k, v in d.get("phase_noise_dbc_hz", {}).items()},
        )

    def __repr__(self) -> str:
        return (
            f"Signal(fc={self.carrier_frequency/1e9:.3f} GHz, "
            f"P={self.power_dbm:.1f} dBm, "
            f"spurs={len(self.spurs)})"
        )
