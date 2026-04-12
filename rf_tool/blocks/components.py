"""
All concrete RF component block implementations.

Each class inherits from RFBlock and overrides:
  - BLOCK_TYPE: string identifier used for serialisation.
  - _setup_ports(): define input/output ports.
  - process(): apply component-specific signal transformation.
  - to_dict() / from_dict(): for JSON serialisation.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

import numpy as np

from rf_tool.models.rf_block import RFBlock, Port
from rf_tool.models.signal import Signal, SpurTone


# ======================================================================= #
# Amplifier / Gain Block                                                   #
# ======================================================================= #

class Amplifier(RFBlock):
    """Standard active amplifier / gain block."""

    BLOCK_TYPE = "Amplifier"

    def __init__(self, **kwargs):
        kwargs.setdefault("label", "Amp")
        kwargs.setdefault("color", "#3A7BD5")
        kwargs.setdefault("gain_db", 15.0)
        kwargs.setdefault("nf_db", 3.0)
        kwargs.setdefault("p1db_dbm", 20.0)
        kwargs.setdefault("oip3_dbm", 30.0)
        super().__init__(**kwargs)


# ======================================================================= #
# Attenuator                                                               #
# ======================================================================= #

class Attenuator(RFBlock):
    """Passive attenuator.  Gain is negative (= attenuation in dB)."""

    BLOCK_TYPE = "Attenuator"

    def __init__(self, attenuation_db: float = 10.0, **kwargs):
        kwargs.setdefault("label", f"{attenuation_db} dB Att")
        kwargs.setdefault("color", "#888888")
        # Attenuator gain = -attenuation
        kwargs["gain_db"] = -abs(attenuation_db)
        # NF of passive attenuator = attenuation (at room temperature)
        kwargs.setdefault("nf_db", abs(attenuation_db))
        self.attenuation_db: float = abs(attenuation_db)
        super().__init__(**kwargs)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["attenuation_db"] = self.attenuation_db
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Attenuator":
        att = abs(d.get("attenuation_db", abs(d.get("gain_db", 10.0))))
        obj = cls(
            attenuation_db=att,
            block_id=d.get("block_id"),
            label=d.get("label", f"{att} dB Att"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#888888"),
            nf_db=d.get("nf_db", att),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


# ======================================================================= #
# Mixer                                                                    #
# ======================================================================= #

class Mixer(RFBlock):
    """
    Frequency-converting mixer.

    Ports: RF (input), LO (input), IF (output).

    The output frequency components are defined by an expression list
    such as "RF-LO", "RF+LO" etc.  The default is single-sideband
    down-conversion: IF = RF - LO.

    The conversion gain (or loss) is specified in gain_db.
    """

    BLOCK_TYPE = "Mixer"

    def __init__(
        self,
        conversion_expressions: Optional[List[str]] = None,
        **kwargs,
    ):
        kwargs.setdefault("label", "Mixer")
        kwargs.setdefault("color", "#E8A838")
        kwargs.setdefault("gain_db", -7.0)       # typical conversion loss
        kwargs.setdefault("nf_db", 7.0)
        # Default: IF = RF - LO  (down-conversion)
        self.conversion_expressions: List[str] = conversion_expressions or ["RF-LO"]
        self._last_rf_signal: Optional[Signal] = None
        self._last_lo_signal: Optional[Signal] = None
        super().__init__(**kwargs)

    def _setup_ports(self) -> None:
        self._input_ports = [Port("RF", "input", 0), Port("LO", "input", 1)]
        self._output_ports = [Port("IF", "output", 0)]

    def process(self, signal: Signal, port_name: str = "RF") -> Dict[str, Signal]:
        """
        Mix RF and LO inputs.

        Output components are generated from every RF tone × LO tone combination.
        """
        if port_name == "RF":
            self._last_rf_signal = signal.copy()
        elif port_name == "LO":
            self._last_lo_signal = signal.copy()
        else:
            self._last_rf_signal = signal.copy()

        rf_sig = self._last_rf_signal
        lo_sig = self._last_lo_signal
        if rf_sig is None or lo_sig is None:
            return {}

        rf_components = self._all_components(rf_sig)
        lo_components = self._all_components(lo_sig)
        if not rf_components or not lo_components:
            return {}

        coeffs = self._effective_mixing_coefficients()
        tones: List[tuple] = []
        for m, n, rel_db in coeffs:
            for f_rf, p_rf in rf_components:
                for f_lo, p_lo in lo_components:
                    f_out = m * f_rf + n * f_lo
                    p_out = p_rf + p_lo + self.gain_db + rel_db
                    tones.append((f_out, p_out))

        if not tones:
            return {}

        tones.sort(key=lambda x: x[1], reverse=True)
        carrier_f, carrier_p = tones[0]
        out_signal = Signal(carrier_frequency=carrier_f, power_dbm=carrier_p, spurs=[])
        for f_out, p_out in tones[1:]:
            out_signal.add_spur(f_out, p_out)

        rf_nf = rf_sig.get_noise_floor_dbm()
        lo_nf = lo_sig.get_noise_floor_dbm()
        noise_terms = []
        if rf_nf is not None:
            noise_terms.append(10.0 ** ((rf_nf + self.gain_db) / 10.0))
        if lo_nf is not None:
            noise_terms.append(10.0 ** ((lo_nf + self.gain_db) / 10.0))
        if noise_terms:
            total_noise_mw = sum(noise_terms)
            out_signal.set_noise_floor_dbm(10.0 * math.log10(max(total_noise_mw, 1e-300)))

        return {"IF": out_signal}

    @staticmethod
    def _eval_freq_expr(expr: str, f_rf: float, f_lo: float) -> float:
        """Safely evaluate a frequency expression like 'RF-LO' or '2*RF+LO'."""
        # Only allow safe characters
        safe_expr = expr.upper().replace(" ", "")
        allowed = set("0123456789.+-*/RFLO()")
        if not all(c in allowed for c in safe_expr):
            raise ValueError(f"Unsafe expression: {expr!r}")
        result = eval(safe_expr, {"__builtins__": {}}, {"RF": f_rf, "LO": f_lo})  # noqa: S307
        return float(result)

    @classmethod
    def _expr_to_mn(cls, expr: str) -> Optional[tuple[int, int]]:
        """Extract linear coefficients (m, n) from an expression m*RF+n*LO."""
        try:
            m = cls._eval_freq_expr(expr, 1.0, 0.0)
            n = cls._eval_freq_expr(expr, 0.0, 1.0)
            c = cls._eval_freq_expr(expr, 0.0, 0.0)
        except (ValueError, TypeError, SyntaxError, NameError):
            return None
        if abs(c) > 1e-12:
            return None
        m_i = int(round(m))
        n_i = int(round(n))
        if abs(m - m_i) > 1e-9 or abs(n - n_i) > 1e-9:
            return None
        return m_i, n_i

    @staticmethod
    def _all_components(signal: Signal) -> List[tuple]:
        comps = [(signal.carrier_frequency, signal.power_dbm)]
        comps.extend((s.frequency, s.power_dbm) for s in signal.spurs)
        return comps

    def _effective_mixing_coefficients(self) -> List[tuple]:
        """
        Return list of (m, n, rel_power_db) coefficients.

        Uses explicit spur coefficients when provided; otherwise converts
        conversion expressions to linear coefficients.
        """
        coeffs: List[tuple] = []
        for expr in self.conversion_expressions:
            mn = self._expr_to_mn(expr)
            if mn is not None:
                coeffs.append((mn[0], mn[1], 0.0))
        for coeff in self.spur_coefficients:
            m = int(coeff.get("m", 1))
            n = int(coeff.get("n", 0))
            rel_db = float(coeff.get("rel_power_db", 0.0))
            coeffs.append((m, n, rel_db))
        if not coeffs:
            coeffs.append((1, -1, 0.0))
        unique = []
        seen = set()
        for m, n, rel_db in coeffs:
            key = (m, n, round(rel_db, 9))
            if key in seen:
                continue
            seen.add(key)
            unique.append((m, n, rel_db))
        return unique

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["conversion_expressions"] = self.conversion_expressions
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mixer":
        obj = cls(
            conversion_expressions=d.get("conversion_expressions", ["RF-LO"]),
            block_id=d.get("block_id"),
            label=d.get("label", "Mixer"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#E8A838"),
            gain_db=d.get("gain_db", -7.0),
            nf_db=d.get("nf_db", 7.0),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


# ======================================================================= #
# S-Parameter Block                                                        #
# ======================================================================= #

class SparBlock(RFBlock):
    """
    Block whose frequency response is defined by an S-parameter file
    (.s2p, .s3p, .s4p) loaded via scikit-rf.

    Gain is derived from |S21|.  Non-linear parameters (NF, P1dB, OIP3)
    may be manually overridden.  Spurious tone generation is disabled
    (output tone = input tone) because the S-params define linear behavior.
    """

    BLOCK_TYPE = "SparBlock"

    def __init__(
        self,
        spar_file: Optional[str] = None,
        **kwargs,
    ):
        kwargs.setdefault("label", "S-Param")
        kwargs.setdefault("color", "#6A0DAD")
        self.spar_file: Optional[str] = spar_file
        self._network = None   # will hold skrf.Network when loaded
        if spar_file:
            self._load_network(spar_file)
        super().__init__(**kwargs)

    def _load_network(self, path: str) -> None:
        try:
            import skrf
            self._network = skrf.Network(path)
        except Exception as exc:
            self._network = None
            print(f"SparBlock: could not load {path!r}: {exc}")

    def get_gain_db_at(self, freq_hz: float) -> float:
        """Interpolate |S21| dB from loaded network at given frequency."""
        if self._network is None:
            return self.gain_db
        import numpy as np
        freqs = self._network.f
        s21 = self._network.s[:, 1, 0]
        s21_interp = np.interp(freq_hz, freqs, np.abs(s21))
        if s21_interp <= 0:
            return -200.0
        return 20.0 * math.log10(s21_interp)

    def process(self, signal: Signal, port_name: str = "IN") -> Dict[str, Signal]:
        """Apply S21 at the signal's carrier frequency (interpolated)."""
        gain = self.get_gain_db_at(signal.carrier_frequency)
        out = signal.apply_gain(gain)
        # Do NOT generate spurs (S-params = linear block)
        return {"OUT": out}

    def get_network(self):
        """Return the loaded skrf.Network (or None)."""
        return self._network

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["spar_file"] = self.spar_file
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SparBlock":
        obj = cls(
            spar_file=d.get("spar_file"),
            block_id=d.get("block_id"),
            label=d.get("label", "S-Param"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#6A0DAD"),
            gain_db=d.get("gain_db", 0.0),
            nf_db=d.get("nf_db", 0.0),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


# ======================================================================= #
# Transfer Function Block (Laplace domain H(s))                           #
# ======================================================================= #

class TransferFnBlock(RFBlock):
    """
    Block whose frequency response is defined by a Laplace-domain
    transfer function H(s) = num(s) / den(s).

    Coefficients are entered as polynomial coefficient lists
    (highest power first, matching scipy.signal.lti convention).

    Gain in dB is evaluated at the signal's carrier frequency.
    """

    BLOCK_TYPE = "TransferFnBlock"

    def __init__(
        self,
        numerator: Optional[List[float]] = None,
        denominator: Optional[List[float]] = None,
        **kwargs,
    ):
        kwargs.setdefault("label", "H(s)")
        kwargs.setdefault("color", "#20A030")
        self.numerator: List[float] = numerator or [1.0]
        self.denominator: List[float] = denominator or [1.0]
        super().__init__(**kwargs)

    def _get_lti(self):
        from scipy.signal import lti
        return lti(self.numerator, self.denominator)

    def gain_db_at_freq(self, freq_hz: float) -> float:
        """Evaluate |H(j2πf)| in dB."""
        import numpy as np
        omega = 2.0 * math.pi * freq_hz
        sys = self._get_lti()
        _, mag, _ = sys.bode([omega])
        return float(mag[0])

    def process(self, signal: Signal, port_name: str = "IN") -> Dict[str, Signal]:
        gain = self.gain_db_at_freq(signal.carrier_frequency)
        out = signal.apply_gain(gain)
        return {"OUT": out}

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["numerator"] = self.numerator
        d["denominator"] = self.denominator
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TransferFnBlock":
        obj = cls(
            numerator=d.get("numerator", [1.0]),
            denominator=d.get("denominator", [1.0]),
            block_id=d.get("block_id"),
            label=d.get("label", "H(s)"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#20A030"),
            gain_db=d.get("gain_db", 0.0),
            nf_db=d.get("nf_db", 0.0),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


# ======================================================================= #
# Filters                                                                  #
# ======================================================================= #

class LowPassFilter(RFBlock):
    """
    N-th order Butterworth low-pass filter.

    Gain at the carrier is evaluated using scipy.signal.
    """

    BLOCK_TYPE = "LowPassFilter"

    def __init__(
        self,
        order: int = 3,
        cutoff_hz: float = 1e9,
        **kwargs,
    ):
        kwargs.setdefault("label", f"LPF {order}rd order")
        kwargs.setdefault("color", "#009999")
        self.order: int = order
        self.cutoff_hz: float = cutoff_hz
        super().__init__(**kwargs)

    def gain_db_at_freq(self, freq_hz: float) -> float:
        """Evaluate Butterworth LPF insertion loss at frequency."""
        from scipy.signal import butter, freqs
        import numpy as np
        # Normalized frequency
        wn = 2.0 * math.pi * self.cutoff_hz
        w0 = 2.0 * math.pi * freq_hz
        # Butterworth magnitude: |H(jw)| = 1 / sqrt(1 + (w/wn)^(2*n))
        ratio = (w0 / wn) ** (2 * self.order)
        mag = 1.0 / math.sqrt(1.0 + ratio)
        return 20.0 * math.log10(mag) if mag > 0 else -300.0

    def process(self, signal: Signal, port_name: str = "IN") -> Dict[str, Signal]:
        gain = self.gain_db_at_freq(signal.carrier_frequency)
        out = signal.apply_gain(gain)
        return {"OUT": out}

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["order"] = self.order
        d["cutoff_hz"] = self.cutoff_hz
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LowPassFilter":
        obj = cls(
            order=d.get("order", 3),
            cutoff_hz=d.get("cutoff_hz", 1e9),
            block_id=d.get("block_id"),
            label=d.get("label", "LPF"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#009999"),
            gain_db=d.get("gain_db", 0.0),
            nf_db=d.get("nf_db", 0.0),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


class HighPassFilter(RFBlock):
    """N-th order Butterworth high-pass filter."""

    BLOCK_TYPE = "HighPassFilter"

    def __init__(self, order: int = 3, cutoff_hz: float = 1e9, **kwargs):
        kwargs.setdefault("label", f"HPF {order}rd order")
        kwargs.setdefault("color", "#CC6600")
        self.order: int = order
        self.cutoff_hz: float = cutoff_hz
        super().__init__(**kwargs)

    def gain_db_at_freq(self, freq_hz: float) -> float:
        wn = 2.0 * math.pi * self.cutoff_hz
        w0 = 2.0 * math.pi * freq_hz
        if w0 == 0:
            return -300.0
        ratio = (wn / w0) ** (2 * self.order)
        mag = 1.0 / math.sqrt(1.0 + ratio)
        return 20.0 * math.log10(mag) if mag > 0 else -300.0

    def process(self, signal: Signal, port_name: str = "IN") -> Dict[str, Signal]:
        gain = self.gain_db_at_freq(signal.carrier_frequency)
        out = signal.apply_gain(gain)
        return {"OUT": out}

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["order"] = self.order
        d["cutoff_hz"] = self.cutoff_hz
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HighPassFilter":
        obj = cls(
            order=d.get("order", 3),
            cutoff_hz=d.get("cutoff_hz", 1e9),
            block_id=d.get("block_id"),
            label=d.get("label", "HPF"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#CC6600"),
            gain_db=d.get("gain_db", 0.0),
            nf_db=d.get("nf_db", 0.0),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


# ======================================================================= #
# Power Splitter / Combiner                                               #
# ======================================================================= #

class PowerSplitter(RFBlock):
    """
    N-way power splitter or combiner.

    Splitting loss = 10*log10(N) dB applied per output path.
    """

    BLOCK_TYPE = "PowerSplitter"

    def __init__(self, n_ways: int = 2, is_combiner: bool = False, **kwargs):
        self.n_ways: int = max(2, n_ways)
        self.is_combiner: bool = is_combiner
        split_loss_db = -10.0 * math.log10(self.n_ways)
        kwargs.setdefault(
            "label",
            f"{self.n_ways}-way {'Combiner' if is_combiner else 'Splitter'}",
        )
        kwargs.setdefault("color", "#A0A0FF")
        kwargs["gain_db"] = split_loss_db
        kwargs.setdefault("nf_db", abs(split_loss_db))
        super().__init__(**kwargs)
        # For combiners: buffer signals from all inputs to combine them
        self._combine_buffer: Dict[str, Signal] = {}

    def set_n_ways(self, n_ways: int) -> None:
        """Update split/combine ways and rebuild dynamic ports and loss settings."""
        self.n_ways = max(2, n_ways)
        split_loss_db = -10.0 * math.log10(self.n_ways)
        self.gain_db = split_loss_db
        self.nf_db = abs(split_loss_db)
        self._setup_ports()

    def _setup_ports(self) -> None:
        if self.is_combiner:
            self._input_ports = [Port(f"IN{i}", "input", i) for i in range(self.n_ways)]
            self._output_ports = [Port("OUT", "output", 0)]
        else:
            self._input_ports = [Port("IN", "input", 0)]
            self._output_ports = [Port(f"OUT{i}", "output", i) for i in range(self.n_ways)]

    def process(self, signal: Signal, port_name: str = "IN") -> Dict[str, Signal]:
        if self.is_combiner:
            # Store signal from this input port
            self._combine_buffer[port_name] = signal
            
            # Combine all currently buffered signals (process any available inputs, don't wait for all)
            if self._combine_buffer:
                combined = None
                for input_port in sorted(self._combine_buffer.keys()):
                    sig = self._combine_buffer[input_port]
                    if combined is None:
                        combined = sig.copy()
                    else:
                        # Merge signals: add power in linear domain
                        p_total_mw = (10.0 ** (combined.power_dbm / 10.0)) + (10.0 ** (sig.power_dbm / 10.0))
                        combined.power_dbm = 10.0 * math.log10(max(p_total_mw, 1e-300))
                        # Merge spurs
                        for spur in sig.spurs:
                            found = False
                            for c_spur in combined.spurs:
                                if abs(c_spur.frequency - spur.frequency) < 1e-3:
                                    spur_mw = (10.0 ** (c_spur.power_dbm / 10.0)) + (10.0 ** (spur.power_dbm / 10.0))
                                    c_spur.power_dbm = 10.0 * math.log10(max(spur_mw, 1e-300))
                                    found = True
                                    break
                            if not found:
                                combined.spurs.append(spur.copy())
                        # Merge SNR
                        combined_mw = 10.0 ** (combined.power_dbm / 10.0)
                        sig_mw = 10.0 ** (sig.power_dbm / 10.0)
                        if combined.snr_db is not None and sig.snr_db is not None:
                            noise_combined = combined_mw / (10.0 ** (combined.snr_db / 10.0))
                            noise_sig = sig_mw / (10.0 ** (sig.snr_db / 10.0))
                            total_noise = noise_combined + noise_sig
                            total_signal = combined_mw + sig_mw
                            combined.snr_db = 10.0 * math.log10(total_signal / max(total_noise, 1e-300))
                
                # Apply gain to combined signal
                out = combined.apply_gain(self.gain_db)
                return {"OUT": out}
            else:
                return {}
        else:
            out = signal.apply_gain(self.gain_db)
            return {f"OUT{i}": out.copy() for i in range(self.n_ways)}

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["n_ways"] = self.n_ways
        d["is_combiner"] = self.is_combiner
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PowerSplitter":
        obj = cls(
            n_ways=d.get("n_ways", 2),
            is_combiner=d.get("is_combiner", False),
            block_id=d.get("block_id"),
            label=d.get("label", "Splitter"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#A0A0FF"),
            nf_db=d.get("nf_db", 3.01),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


class PowerCombiner(PowerSplitter):
    """Dedicated N-way power combiner block."""

    BLOCK_TYPE = "PowerCombiner"

    def __init__(self, n_ways: int = 2, **kwargs):
        kwargs.setdefault("label", f"{max(2, n_ways)}-way Combiner")
        kwargs.setdefault("color", "#8C8CFF")
        super().__init__(n_ways=n_ways, is_combiner=True, **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "PowerCombiner":
        obj = cls(
            n_ways=d.get("n_ways", 2),
            block_id=d.get("block_id"),
            label=d.get("label", "2-way Combiner"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#8C8CFF"),
            nf_db=d.get("nf_db", 3.01),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


# ======================================================================= #
# Switch (1x2 and 2x1)                                                     #
# ======================================================================= #

class Switch(RFBlock):
    """
    RF switch with configurable topology (1x2 or 2x1).

    The active port is toggled via toggle_state() or by setting
    active_port.

    Insertion loss and isolation are configurable.
    """

    BLOCK_TYPE = "Switch"

    def __init__(
        self,
        topology: str = "1x2",
        active_port: int = 0,
        insertion_loss_db: float = 0.5,
        isolation_db: float = 40.0,
        **kwargs,
    ):
        self.topology: str = topology     # "1x2" or "2x1"
        self.active_port: int = active_port
        self.insertion_loss_db: float = insertion_loss_db
        self.isolation_db: float = isolation_db
        kwargs.setdefault("label", f"SW {topology}")
        kwargs.setdefault("color", "#DD4444")
        kwargs["gain_db"] = -insertion_loss_db
        kwargs.setdefault("nf_db", insertion_loss_db)
        super().__init__(**kwargs)

    def _setup_ports(self) -> None:
        if self.topology == "1x2":
            self._input_ports = [Port("IN", "input", 0)]
            self._output_ports = [Port("OUT0", "output", 0), Port("OUT1", "output", 1)]
        else:  # 2x1
            self._input_ports = [Port("IN0", "input", 0), Port("IN1", "input", 1)]
            self._output_ports = [Port("OUT", "output", 0)]

    def toggle_state(self) -> None:
        """Toggle the active port."""
        n_ports = 2  # only supporting 1x2 or 2x1
        self.active_port = (self.active_port + 1) % n_ports

    def process(self, signal: Signal, port_name: str = "IN") -> Dict[str, Signal]:
        if self.topology == "1x2":
            out_active = signal.apply_gain(-self.insertion_loss_db)
            out_isolated = signal.apply_gain(-self.isolation_db)
            outputs = {}
            outputs["OUT0"] = out_active if self.active_port == 0 else out_isolated
            outputs["OUT1"] = out_isolated if self.active_port == 0 else out_active
            return outputs
        else:
            # 2x1: route active input to output
            if port_name in (f"IN{self.active_port}", "IN0"):
                out = signal.apply_gain(-self.insertion_loss_db)
            else:
                out = signal.apply_gain(-self.isolation_db)
            return {"OUT": out}

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["topology"] = self.topology
        d["active_port"] = self.active_port
        d["insertion_loss_db"] = self.insertion_loss_db
        d["isolation_db"] = self.isolation_db
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Switch":
        obj = cls(
            topology=d.get("topology", "1x2"),
            active_port=d.get("active_port", 0),
            insertion_loss_db=d.get("insertion_loss_db", 0.5),
            isolation_db=d.get("isolation_db", 40.0),
            block_id=d.get("block_id"),
            label=d.get("label", "Switch"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#DD4444"),
            nf_db=d.get("nf_db", 0.5),
            p1db_dbm=d.get("p1db_dbm"),
            oip3_dbm=d.get("oip3_dbm"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            spur_coefficients=d.get("spur_coefficients", []),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


# ======================================================================= #
# Source                                                                   #
# ======================================================================= #

class Source(RFBlock):
    """
    Signal source (e.g. local oscillator, signal generator).
    Generates a Signal with specified frequency and power.
    Has no input port.
    """

    BLOCK_TYPE = "Source"

    def __init__(
        self,
        frequency: float = 1e9,
        output_power_dbm: float = 0.0,
        snr_db: Optional[float] = None,
        **kwargs,
    ):
        self.frequency: float = frequency
        self.output_power_dbm: float = output_power_dbm
        self.snr_db: Optional[float] = snr_db
        kwargs.setdefault("label", "Source")
        kwargs.setdefault("color", "#33CC33")
        kwargs["gain_db"] = 0.0
        kwargs["nf_db"] = 0.0
        super().__init__(**kwargs)

    def _setup_ports(self) -> None:
        self._input_ports = []
        self._output_ports = [Port("OUT", "output", 0)]

    def generate(self) -> Signal:
        """Create and return the source signal."""
        sig = Signal(
            carrier_frequency=self.frequency,
            power_dbm=self.output_power_dbm,
            snr_db=self.snr_db,
        )
        if self.snr_db is not None:
            sig.set_noise_floor_dbm(self.output_power_dbm - self.snr_db)
        return sig

    def process(self, signal=None, port_name: str = "OUT") -> Dict[str, Signal]:
        return {"OUT": self.generate()}

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["frequency"] = self.frequency
        d["output_power_dbm"] = self.output_power_dbm
        d["snr_db"] = self.snr_db
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        obj = cls(
            frequency=d.get("frequency", 1e9),
            output_power_dbm=d.get("output_power_dbm", 0.0),
            snr_db=d.get("snr_db"),
            block_id=d.get("block_id"),
            label=d.get("label", "Source"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#33CC33"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


# ======================================================================= #
# Sink (Termination)                                                       #
# ======================================================================= #

class Sink(RFBlock):
    """
    Signal sink / termination.
    Captures the arriving signal for display.
    Has no output port.
    Double-clicking opens a spectrum plot.
    """

    BLOCK_TYPE = "Sink"

    def __init__(self, **kwargs):
        kwargs.setdefault("label", "Sink")
        kwargs.setdefault("color", "#CC3333")
        kwargs["gain_db"] = 0.0
        kwargs["nf_db"] = 0.0
        self.last_signal: Optional[Signal] = None
        super().__init__(**kwargs)

    def _setup_ports(self) -> None:
        self._input_ports = [Port("IN", "input", 0)]
        self._output_ports = []

    def process(self, signal: Signal, port_name: str = "IN") -> Dict[str, Signal]:
        self.last_signal = signal.copy()
        return {}

    @classmethod
    def from_dict(cls, d: dict) -> "Sink":
        obj = cls(
            block_id=d.get("block_id"),
            label=d.get("label", "Sink"),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
            color=d.get("color", "#CC3333"),
            min_input_power_dbm=d.get("min_input_power_dbm"),
            max_input_power_dbm=d.get("max_input_power_dbm"),
            comment_mode=d.get("comment_mode", "active"),
        )
        return obj


# ======================================================================= #
# Block registry                                                           #
# ======================================================================= #

from rf_tool.blocks.hierarchical import HierInputPin, HierOutputPin, HierSubcircuit  # noqa: E402

BLOCK_REGISTRY: Dict[str, type] = {
    "RFBlock": RFBlock,
    "Amplifier": Amplifier,
    "Attenuator": Attenuator,
    "Mixer": Mixer,
    "SparBlock": SparBlock,
    "TransferFnBlock": TransferFnBlock,
    "LowPassFilter": LowPassFilter,
    "HighPassFilter": HighPassFilter,
    "PowerSplitter": PowerSplitter,
    "PowerCombiner": PowerCombiner,
    "Switch": Switch,
    "Source": Source,
    "Sink": Sink,
    "HierInputPin": HierInputPin,
    "HierOutputPin": HierOutputPin,
    "HierSubcircuit": HierSubcircuit,
}


def block_from_dict(d: dict) -> RFBlock:
    """Deserialise a block from a dict, dispatching to the correct subclass."""
    block_type = d.get("block_type", "RFBlock")
    cls = BLOCK_REGISTRY.get(block_type, RFBlock)
    return cls.from_dict(d)
