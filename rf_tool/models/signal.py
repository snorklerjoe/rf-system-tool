"""
Signal data model for RF System Tool.

A Signal carries carrier frequency, power level, spurious tones,
and phase noise parameters through the RF signal chain.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import math


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
    snr_db : float or None
        Signal-to-noise ratio carried with this signal.
    """
    carrier_frequency: float          # Hz
    power_dbm: float                  # dBm
    spurs: List[SpurTone] = field(default_factory=list)
    phase_noise_dbc_hz: Dict[float, float] = field(default_factory=dict)
    snr_db: Optional[float] = None
    noise_floor_dbm: Optional[float] = None

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
            snr_db=self.snr_db,
            noise_floor_dbm=self.noise_floor_dbm,
        )

    # ------------------------------------------------------------------ #
    # Propagation helpers                                                  #
    # ------------------------------------------------------------------ #
    def apply_gain(self, gain_db: float) -> "Signal":
        """Return a new Signal with gain applied to carrier and all spurs."""
        new_sig = self.copy()
        new_sig.power_dbm += gain_db
        new_sig.spurs = [s.scale_power(gain_db) for s in new_sig.spurs]
        if new_sig.noise_floor_dbm is not None:
            new_sig.noise_floor_dbm += gain_db
        if new_sig.snr_db is None and new_sig.noise_floor_dbm is not None:
            new_sig.snr_db = new_sig.power_dbm - new_sig.noise_floor_dbm
        return new_sig

    def total_power_dbm(self) -> float:
        """Return total signal power (carrier + spurs) in dBm."""
        tones = [(self.carrier_frequency, self.power_dbm)]
        tones.extend((s.frequency, s.power_dbm) for s in self.spurs)
        total_mw = sum(10.0 ** (p_dbm / 10.0) for _f_hz, p_dbm in tones)
        return 10.0 * math.log10(max(total_mw, 1e-300))

    @staticmethod
    def _combined_tones(tones: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Combine same-frequency tones in linear power and return sorted tones."""
        bins: List[Tuple[float, float]] = []
        for freq_hz, power_dbm in tones:
            power_mw = 10.0 ** (power_dbm / 10.0)
            merged = False
            for idx, (f_bin, p_bin_mw) in enumerate(bins):
                if abs(f_bin - freq_hz) < 1e-3:
                    bins[idx] = (f_bin, p_bin_mw + power_mw)
                    merged = True
                    break
            if not merged:
                bins.append((freq_hz, power_mw))
        out = [(f_hz, 10.0 * math.log10(max(p_mw, 1e-300))) for f_hz, p_mw in bins]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def apply_frequency_response(self, gain_fn: Callable[[float], float]) -> "Signal":
        """
        Apply frequency-dependent gain to each tone independently.

        The output carrier is chosen as the strongest resulting tone.
        """
        tones = [(self.carrier_frequency, self.power_dbm)]
        tones.extend((s.frequency, s.power_dbm) for s in self.spurs)
        gained_tones = [(f_hz, p_dbm + gain_fn(f_hz)) for f_hz, p_dbm in tones]
        combined = self._combined_tones(gained_tones)
        if not combined:
            return self.copy()

        carrier_f, carrier_p = combined[0]
        out = Signal(carrier_frequency=carrier_f, power_dbm=carrier_p, spurs=[])
        for f_hz, p_dbm in combined[1:]:
            out.add_spur(f_hz, p_dbm)

        in_noise = self.get_noise_floor_dbm()
        if in_noise is not None:
            carrier_gain = gain_fn(self.carrier_frequency)
            out.set_noise_floor_dbm(in_noise + carrier_gain)
        else:
            out.snr_db = self.snr_db
        return out

    def get_noise_floor_dbm(self) -> Optional[float]:
        """Return explicit noise floor if present, else infer from carrier SNR."""
        if self.noise_floor_dbm is not None:
            return self.noise_floor_dbm
        if self.snr_db is None:
            return None
        return self.power_dbm - self.snr_db

    def set_noise_floor_dbm(self, noise_floor_dbm: Optional[float]) -> None:
        """Set noise floor and keep SNR consistent with carrier power."""
        self.noise_floor_dbm = noise_floor_dbm
        if noise_floor_dbm is None:
            return
        self.snr_db = self.power_dbm - noise_floor_dbm if math.isfinite(noise_floor_dbm) else None

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
            "snr_db": self.snr_db,
            "noise_floor_dbm": self.noise_floor_dbm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Signal":
        return cls(
            carrier_frequency=d["carrier_frequency"],
            power_dbm=d["power_dbm"],
            spurs=[SpurTone.from_dict(s) for s in d.get("spurs", [])],
            phase_noise_dbc_hz={float(k): v for k, v in d.get("phase_noise_dbc_hz", {}).items()},
            snr_db=d.get("snr_db"),
            noise_floor_dbm=d.get("noise_floor_dbm"),
        )

    def __repr__(self) -> str:
        return (
            f"Signal(fc={self.carrier_frequency/1e9:.3f} GHz, "
            f"P={self.power_dbm:.1f} dBm, "
            f"spurs={len(self.spurs)})"
        )
