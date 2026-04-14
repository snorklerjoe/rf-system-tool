"""
Unit tests for all RF block component classes.

Covers per-block:
  - Construction and defaults
  - Port configuration
  - process() output: carrier frequency, power, spurs
  - Serialisation round-trip (to_dict / from_dict)
  - Power-check logic (check_power)
  - Derived quantities (iip3_dbm, oip3_dbm_inferred, gain_linear, nf_linear)

Also covers:
  - block_from_dict dispatcher
  - BLOCK_REGISTRY
  - Filter math (LPF / HPF gain at dc, Nyquist, cutoff)
  - Mixer frequency conversion expressions
  - Switch toggling
  - Source.generate()
  - Sink.process() captures signal
"""
import json
import math
import pytest
import numpy as np

from rf_tool.models.signal import Signal, SpurTone
from rf_tool.models.rf_block import RFBlock, Port
from rf_tool.blocks.components import (
    Amplifier, Attenuator, Mixer, SparBlock, TransferFnBlock,
    LowPassFilter, HighPassFilter, PowerSplitter, PowerCombiner, Switch, Source, Sink,
    BLOCK_REGISTRY, block_from_dict,
)
from rf_tool.blocks.hierarchical import HierSubcircuit, analysis_blocks_from_subcircuit


# ======================================================================= #
# Helpers                                                                  #
# ======================================================================= #

def make_signal(fc=1e9, pwr=-10.0, spurs=None):
    spurs = spurs or []
    return Signal(carrier_frequency=fc, power_dbm=pwr, spurs=spurs)


# ======================================================================= #
# RFBlock base class                                                       #
# ======================================================================= #

class TestRFBlock:
    def test_default_ports(self):
        b = RFBlock()
        assert len(b.input_ports) == 1
        assert len(b.output_ports) == 1
        assert b.input_ports[0].direction == "input"
        assert b.output_ports[0].direction == "output"

    def test_unique_ids(self):
        b1 = RFBlock()
        b2 = RFBlock()
        assert b1.block_id != b2.block_id

    def test_explicit_id_preserved(self):
        b = RFBlock(block_id="test-id-123")
        assert b.block_id == "test-id-123"

    def test_gain_linear_conversion(self):
        b = RFBlock(gain_db=20.0)
        assert b.gain_linear == pytest.approx(100.0, rel=1e-9)

    def test_nf_linear_conversion(self):
        b = RFBlock(nf_db=3.0103)
        assert b.nf_linear == pytest.approx(2.0, rel=1e-4)

    def test_oip3_inferred_from_p1db(self):
        b = RFBlock(p1db_dbm=20.0)
        assert b.oip3_dbm_inferred == pytest.approx(29.6, abs=1e-6)

    def test_oip3_direct_takes_priority(self):
        b = RFBlock(p1db_dbm=20.0, oip3_dbm=35.0)
        assert b.oip3_dbm_inferred == pytest.approx(35.0)

    def test_iip3_from_oip3_and_gain(self):
        b = RFBlock(gain_db=15.0, oip3_dbm=30.0)
        assert b.iip3_dbm == pytest.approx(15.0, abs=1e-6)

    def test_iip3_none_when_no_nonlinear(self):
        b = RFBlock(gain_db=15.0)
        assert b.iip3_dbm is None

    # check_power
    def test_check_power_ok(self):
        b = RFBlock(min_input_power_dbm=-30.0, max_input_power_dbm=10.0)
        assert b.check_power(0.0) == "ok"

    def test_check_power_high(self):
        b = RFBlock(max_input_power_dbm=10.0)
        assert b.check_power(15.0) == "high"

    def test_check_power_low(self):
        b = RFBlock(min_input_power_dbm=-20.0)
        assert b.check_power(-30.0) == "low"

    def test_check_power_no_limits_always_ok(self):
        b = RFBlock()
        assert b.check_power(-200.0) == "ok"
        assert b.check_power(200.0) == "ok"

    # process default
    def test_process_applies_gain(self):
        b = RFBlock(gain_db=10.0)
        sig = make_signal(pwr=-20.0)
        out = b.process(sig)
        assert out["OUT"].power_dbm == pytest.approx(-10.0)

    def test_process_frequency_unchanged(self):
        b = RFBlock(gain_db=10.0)
        sig = make_signal(fc=2.4e9)
        out = b.process(sig)
        assert out["OUT"].carrier_frequency == pytest.approx(2.4e9)

    # serialisation
    def test_serialisation_round_trip(self):
        b = RFBlock(
            label="Test", x=100.0, y=200.0, gain_db=15.0, nf_db=3.0,
            p1db_dbm=20.0, oip3_dbm=30.0,
            min_input_power_dbm=-30.0, max_input_power_dbm=10.0,
            spur_coefficients=[{"m": 2, "n": 0, "rel_power_db": -30.0}],
            comment_mode="through",
        )
        b2 = RFBlock.from_dict(b.to_dict())
        assert b2.label == "Test"
        assert b2.x == pytest.approx(100.0)
        assert b2.gain_db == pytest.approx(15.0)
        assert b2.nf_db == pytest.approx(3.0)
        assert b2.oip3_dbm == pytest.approx(30.0)
        assert len(b2.spur_coefficients) == 1
        assert b2.comment_mode == "through"


# ======================================================================= #
# Amplifier                                                                #
# ======================================================================= #

class TestAmplifier:
    def test_defaults(self):
        a = Amplifier()
        assert a.gain_db == pytest.approx(15.0)
        assert a.nf_db == pytest.approx(3.0)
        assert a.p1db_dbm == pytest.approx(20.0)
        assert a.oip3_dbm == pytest.approx(30.0)

    def test_process_gain_and_spur(self):
        a = Amplifier(gain_db=20.0)
        a.spur_coefficients = [{"m": 3, "n": 0, "rel_power_db": -40.0}]
        sig = make_signal(fc=1e9, pwr=-30.0)
        out = a.process(sig)["OUT"]
        assert out.power_dbm == pytest.approx(-10.0)
        assert len(out.spurs) == 1
        assert out.spurs[0].frequency == pytest.approx(3e9)
        assert out.spurs[0].power_dbm == pytest.approx(-10.0 - 40.0)

    def test_block_type(self):
        assert Amplifier.BLOCK_TYPE == "Amplifier"

    def test_round_trip(self):
        a = Amplifier(gain_db=12.0, nf_db=2.5, oip3_dbm=28.0)
        a2 = Amplifier.from_dict(a.to_dict())
        assert a2.gain_db == pytest.approx(12.0)
        assert a2.nf_db == pytest.approx(2.5)
        assert a2.oip3_dbm == pytest.approx(28.0)


# ======================================================================= #
# Attenuator                                                               #
# ======================================================================= #

class TestAttenuator:
    def test_gain_is_negative_attenuation(self):
        a = Attenuator(attenuation_db=10.0)
        assert a.gain_db == pytest.approx(-10.0)

    def test_nf_equals_attenuation_by_default(self):
        a = Attenuator(attenuation_db=6.0)
        assert a.nf_db == pytest.approx(6.0)

    def test_process_reduces_power(self):
        a = Attenuator(attenuation_db=10.0)
        sig = make_signal(pwr=0.0)
        out = a.process(sig)["OUT"]
        assert out.power_dbm == pytest.approx(-10.0)

    def test_process_reduces_spurs(self):
        a = Attenuator(attenuation_db=10.0)
        sig = make_signal(pwr=0.0, spurs=[SpurTone(2e9, -20.0)])
        out = a.process(sig)["OUT"]
        assert out.spurs[0].power_dbm == pytest.approx(-30.0)

    def test_round_trip(self):
        a = Attenuator(attenuation_db=15.0)
        a2 = Attenuator.from_dict(a.to_dict())
        assert a2.attenuation_db == pytest.approx(15.0)
        assert a2.gain_db == pytest.approx(-15.0)
        assert a2.nf_db == pytest.approx(15.0)


# ======================================================================= #
# Mixer                                                                    #
# ======================================================================= #

class TestMixer:
    def test_default_ports(self):
        m = Mixer()
        port_names = [p.name for p in m.input_ports]
        assert "RF" in port_names
        assert "LO" in port_names
        assert len(m.output_ports) == 1
        assert m.output_ports[0].name == "IF"

    def test_down_conversion_frequency(self):
        """RF=2 GHz, LO=1.9 GHz -> IF=0.1 GHz"""
        m = Mixer(conversion_expressions=["RF-LO"])
        m.process(make_signal(fc=2e9, pwr=0.0), "RF")
        out = m.process(make_signal(fc=1.9e9, pwr=0.0), "LO")["IF"]
        assert out.carrier_frequency == pytest.approx(1e8, rel=1e-9)

    def test_up_conversion_frequency(self):
        m = Mixer(conversion_expressions=["RF+LO"])
        m.process(make_signal(fc=100e6, pwr=0.0), "RF")
        out = m.process(make_signal(fc=1e9, pwr=0.0), "LO")["IF"]
        assert out.carrier_frequency == pytest.approx(1.1e9, rel=1e-9)

    def test_conversion_loss_applied(self):
        m = Mixer(gain_db=-7.0, conversion_expressions=["RF-LO"])
        m.process(make_signal(fc=2e9, pwr=0.0), "RF")
        out = m.process(make_signal(fc=1e9, pwr=0.0), "LO")["IF"]
        assert out.power_dbm == pytest.approx(-7.0)

    def test_power_depends_on_rf_and_lo_levels(self):
        m = Mixer(gain_db=-7.0, conversion_expressions=["RF-LO"])
        m.process(make_signal(fc=2e9, pwr=-10.0), "RF")
        out = m.process(make_signal(fc=1e9, pwr=5.0), "LO")["IF"]
        assert out.power_dbm == pytest.approx(-12.0)

    def test_uses_all_rf_lo_component_combinations(self):
        m = Mixer(gain_db=0.0, conversion_expressions=["RF-LO"])
        rf = make_signal(fc=2e9, pwr=0.0, spurs=[SpurTone(2.2e9, -20.0)])
        lo = make_signal(fc=1e9, pwr=0.0, spurs=[SpurTone(1.1e9, -20.0)])
        m.process(rf, "RF")
        out = m.process(lo, "LO")["IF"]
        freqs = [out.carrier_frequency] + [s.frequency for s in out.spurs]
        assert any(abs(f - 1.0e9) < 1e-3 for f in freqs)   # 2.0 - 1.0
        assert any(abs(f - 0.9e9) < 1e-3 for f in freqs)   # 2.0 - 1.1
        assert any(abs(f - 1.2e9) < 1e-3 for f in freqs)   # 2.2 - 1.0
        assert any(abs(f - 1.1e9) < 1e-3 for f in freqs)   # 2.2 - 1.1

    def test_eval_freq_expr_simple(self):
        assert Mixer._eval_freq_expr("RF-LO", 2e9, 1e9) == pytest.approx(1e9)
        assert Mixer._eval_freq_expr("RF+LO", 2e9, 1e9) == pytest.approx(3e9)
        assert Mixer._eval_freq_expr("2*RF-LO", 2e9, 1e9) == pytest.approx(3e9)

    def test_eval_freq_expr_rejects_unsafe(self):
        with pytest.raises(ValueError, match="Unsafe"):
            Mixer._eval_freq_expr("__import__('os')", 1e9, 1e9)

    def test_spur_generation_in_mixer(self):
        m = Mixer(gain_db=0.0)
        m.spur_coefficients = [{"m": 2, "n": 1, "rel_power_db": -30.0}]
        m.process(make_signal(fc=2e9, pwr=0.0), "RF")
        out = m.process(make_signal(fc=1e9, pwr=0.0), "LO")["IF"]
        # Spur at 2*RF + 1*LO = 5 GHz  (m=2, n=1)
        freqs = [out.carrier_frequency] + [s.frequency for s in out.spurs]
        assert any(abs(f - 5e9) < 1e-3 for f in freqs)

    def test_round_trip(self):
        m = Mixer(conversion_expressions=["RF-LO"], gain_db=-6.0)
        m2 = Mixer.from_dict(m.to_dict())
        assert m2.conversion_expressions == ["RF-LO"]
        assert m2.gain_db == pytest.approx(-6.0)


# ======================================================================= #
# SparBlock                                                                #
# ======================================================================= #

class TestSparBlock:
    def test_no_file_uses_scalar_gain(self):
        b = SparBlock(gain_db=5.0)
        sig = make_signal(pwr=0.0)
        out = b.process(sig)["OUT"]
        # Falls back to scalar gain
        assert out.power_dbm == pytest.approx(5.0)

    def test_no_spurs_generated(self):
        """SparBlock is linear and must never generate spurious tones."""
        b = SparBlock(gain_db=5.0)
        b.spur_coefficients = [{"m": 3, "n": 0, "rel_power_db": -30.0}]
        sig = make_signal(pwr=0.0)
        out = b.process(sig)["OUT"]
        assert len(out.spurs) == 0

    def test_no_file_get_gain_returns_scalar(self):
        b = SparBlock(gain_db=-3.0)
        assert b.get_gain_db_at(1e9) == pytest.approx(-3.0)

    def test_round_trip_no_file(self):
        b = SparBlock(gain_db=2.0, nf_db=1.5, spar_file=None)
        b2 = SparBlock.from_dict(b.to_dict())
        assert b2.gain_db == pytest.approx(2.0)
        assert b2.nf_db == pytest.approx(1.5)
        assert b2.spar_file is None

    def test_network_interpolation_with_synthetic_data(self):
        """Build a synthetic Network in memory and verify interpolation."""
        import skrf
        freqs = np.linspace(1e9, 10e9, 101)
        s = np.zeros((101, 2, 2), dtype=complex)
        # S21 = 0.5 (= -6.02 dB) everywhere
        s[:, 1, 0] = 0.5
        freq_obj = skrf.Frequency.from_f(freqs, unit="hz")
        net = skrf.Network(frequency=freq_obj, s=s)

        b = SparBlock()
        b._network = net
        gain = b.get_gain_db_at(5e9)
        assert gain == pytest.approx(20 * math.log10(0.5), abs=1e-4)

    def test_process_applies_frequency_dependent_gain_to_spurs(self):
        b = SparBlock()
        b.get_gain_db_at = lambda f_hz: 10.0 if f_hz < 1.5e9 else -10.0
        sig = make_signal(fc=1e9, pwr=0.0, spurs=[SpurTone(2e9, 0.0)])
        out = b.process(sig)["OUT"]
        assert out.power_dbm == pytest.approx(10.0)
        assert len(out.spurs) == 1
        assert out.spurs[0].frequency == pytest.approx(2e9)
        assert out.spurs[0].power_dbm == pytest.approx(-10.0)


# ======================================================================= #
# TransferFnBlock                                                          #
# ======================================================================= #

class TestTransferFnBlock:
    def test_unity_tf_zero_gain(self):
        """H(s)=1/1 should give 0 dB gain at all frequencies."""
        b = TransferFnBlock(numerator=[1.0], denominator=[1.0])
        # At DC-equivalent: w=0 H(j0)=1 => 0 dB
        gain = b.gain_db_at_freq(0.0 + 1e-9)  # avoid exact 0
        assert gain == pytest.approx(0.0, abs=0.1)

    def test_first_order_lp_rolloff(self):
        """H(s) = wc/(s+wc) - 3 dB at cutoff."""
        fc = 1e6  # 1 MHz
        wc = 2 * math.pi * fc
        # H(s) = wc / (s + wc)  => num=[wc], den=[1, wc]
        b = TransferFnBlock(numerator=[wc], denominator=[1.0, wc])
        gain = b.gain_db_at_freq(fc)
        assert gain == pytest.approx(-3.0103, abs=0.02)

    def test_gain_at_dc_passband(self):
        """H(s) = 10/(s+1) at very low freq -> gain ≈ 20 dB."""
        b = TransferFnBlock(numerator=[10.0], denominator=[1.0, 1.0])
        # At low freq (w->0): |H(j0)| = 10/1 = 10 => 20 dB
        gain = b.gain_db_at_freq(1e-6 / (2 * math.pi))
        assert gain == pytest.approx(20.0, abs=0.5)

    def test_process_returns_out(self):
        b = TransferFnBlock(numerator=[1.0], denominator=[1.0])
        sig = make_signal(fc=1e6, pwr=-10.0)
        out = b.process(sig)
        assert "OUT" in out

    def test_round_trip(self):
        b = TransferFnBlock(numerator=[1.0, 2.0], denominator=[1.0, 3.0, 4.0])
        b2 = TransferFnBlock.from_dict(b.to_dict())
        assert b2.numerator == pytest.approx([1.0, 2.0])
        assert b2.denominator == pytest.approx([1.0, 3.0, 4.0])


# ======================================================================= #
# LowPassFilter                                                            #
# ======================================================================= #

class TestLowPassFilter:
    def test_passband_near_zero_insertion_loss(self):
        """Well below cutoff: gain ≈ 0 dB."""
        f = LowPassFilter(order=3, cutoff_hz=1e9)
        gain = f.gain_db_at_freq(1e6)  # 1 MHz (well below 1 GHz cutoff)
        assert gain == pytest.approx(0.0, abs=0.1)

    def test_stopband_high_attenuation(self):
        """Well above cutoff: significant attenuation."""
        f = LowPassFilter(order=5, cutoff_hz=1e9)
        gain = f.gain_db_at_freq(10e9)  # 10x cutoff
        assert gain < -50.0

    def test_at_cutoff_minus3db(self):
        """At the cutoff frequency: gain = -3.01 dB (Butterworth definition)."""
        f = LowPassFilter(order=1, cutoff_hz=1e9)
        gain = f.gain_db_at_freq(1e9)
        assert gain == pytest.approx(-3.0103, abs=0.01)

    def test_higher_order_steeper_rolloff(self):
        """Higher-order filter has steeper stopband rolloff."""
        f3 = LowPassFilter(order=3, cutoff_hz=1e9)
        f7 = LowPassFilter(order=7, cutoff_hz=1e9)
        freq = 2e9  # one octave above cutoff
        assert f7.gain_db_at_freq(freq) < f3.gain_db_at_freq(freq)

    def test_process_applies_lpf_gain(self):
        f = LowPassFilter(order=3, cutoff_hz=1e9)
        sig = make_signal(fc=10e9, pwr=0.0)  # well above cutoff
        out = f.process(sig)["OUT"]
        assert out.power_dbm < -30.0  # deep in stopband

    def test_process_applies_lpf_gain_per_tone(self):
        f = LowPassFilter(order=4, cutoff_hz=1e9)
        sig = make_signal(fc=100e6, pwr=0.0, spurs=[SpurTone(10e9, 0.0)])
        out = f.process(sig)["OUT"]
        weakest_tone_power = min([out.power_dbm] + [s.power_dbm for s in out.spurs])
        assert out.power_dbm > -1.0  # carrier near passband
        assert weakest_tone_power < -60.0  # high-frequency spur is strongly attenuated

    def test_round_trip(self):
        f = LowPassFilter(order=5, cutoff_hz=2.4e9)
        f2 = LowPassFilter.from_dict(f.to_dict())
        assert f2.order == 5
        assert f2.cutoff_hz == pytest.approx(2.4e9)

    def test_butterworth_magnitude_formula(self):
        """Verify our implementation matches the analytical formula at arbitrary freq."""
        order, fc = 4, 500e6
        f = LowPassFilter(order=order, cutoff_hz=fc)
        test_freq = 750e6
        gain_impl = f.gain_db_at_freq(test_freq)
        # Analytical: |H| = 1/sqrt(1+(f/fc)^(2*n))
        ratio = (test_freq / fc) ** (2 * order)
        gain_theory = 20.0 * math.log10(1.0 / math.sqrt(1.0 + ratio))
        assert gain_impl == pytest.approx(gain_theory, abs=1e-9)


# ======================================================================= #
# HighPassFilter                                                           #
# ======================================================================= #

class TestHighPassFilter:
    def test_stopband_near_dc_high_attenuation(self):
        f = HighPassFilter(order=3, cutoff_hz=1e9)
        gain = f.gain_db_at_freq(1e6)  # well below cutoff
        assert gain < -50.0

    def test_passband_above_cutoff_near_zero(self):
        f = HighPassFilter(order=3, cutoff_hz=1e9)
        gain = f.gain_db_at_freq(100e9)  # well above cutoff
        assert gain == pytest.approx(0.0, abs=0.1)

    def test_at_cutoff_minus3db(self):
        f = HighPassFilter(order=1, cutoff_hz=1e9)
        gain = f.gain_db_at_freq(1e9)
        assert gain == pytest.approx(-3.0103, abs=0.01)

    def test_zero_frequency_returns_large_attenuation(self):
        f = HighPassFilter(order=3, cutoff_hz=1e9)
        gain = f.gain_db_at_freq(0.0)
        assert gain < -200.0

    def test_butterworth_magnitude_formula(self):
        """Verify HPF analytical formula at arbitrary freq."""
        order, fc = 3, 1e9
        f = HighPassFilter(order=order, cutoff_hz=fc)
        test_freq = 500e6
        gain_impl = f.gain_db_at_freq(test_freq)
        ratio = (fc / test_freq) ** (2 * order)
        gain_theory = 20.0 * math.log10(1.0 / math.sqrt(1.0 + ratio))
        assert gain_impl == pytest.approx(gain_theory, abs=1e-9)

    def test_round_trip(self):
        f = HighPassFilter(order=4, cutoff_hz=500e6)
        f2 = HighPassFilter.from_dict(f.to_dict())
        assert f2.order == 4
        assert f2.cutoff_hz == pytest.approx(500e6)

    def test_lpf_and_hpf_complementary_at_cutoff(self):
        """LPF and HPF of same order and cutoff should both be -3dB at fc."""
        fc = 2e9
        lpf = LowPassFilter(order=2, cutoff_hz=fc)
        hpf = HighPassFilter(order=2, cutoff_hz=fc)
        assert lpf.gain_db_at_freq(fc) == pytest.approx(hpf.gain_db_at_freq(fc), abs=1e-6)


# ======================================================================= #
# PowerSplitter                                                            #
# ======================================================================= #

class TestPowerSplitter:
    def test_2way_splitter_loss(self):
        s = PowerSplitter(n_ways=2)
        assert s.gain_db == pytest.approx(-10 * math.log10(2), rel=1e-6)

    def test_4way_splitter_loss(self):
        s = PowerSplitter(n_ways=4)
        assert s.gain_db == pytest.approx(-6.0206, abs=0.001)

    def test_splitter_has_n_outputs(self):
        s = PowerSplitter(n_ways=3)
        assert len(s.output_ports) == 3

    def test_combiner_has_n_inputs(self):
        c = PowerSplitter(n_ways=4, is_combiner=True)
        assert len(c.input_ports) == 4
        assert len(c.output_ports) == 1

    def test_splitter_output_powers_equal(self):
        s = PowerSplitter(n_ways=2)
        sig = make_signal(pwr=0.0)
        out = s.process(sig)
        assert "OUT0" in out
        assert "OUT1" in out
        assert out["OUT0"].power_dbm == pytest.approx(out["OUT1"].power_dbm, abs=1e-9)

    def test_splitter_outputs_are_independent_copies(self):
        """Mutating one output must not affect the other."""
        s = PowerSplitter(n_ways=2)
        sig = make_signal(pwr=0.0)
        out = s.process(sig)
        out["OUT0"].power_dbm = 999.0
        assert out["OUT1"].power_dbm != pytest.approx(999.0)

    def test_round_trip(self):
        s = PowerSplitter(n_ways=3, is_combiner=False)
        s2 = PowerSplitter.from_dict(s.to_dict())
        assert s2.n_ways == 3
        assert s2.is_combiner is False

    def test_set_n_ways_updates_ports(self):
        s = PowerSplitter(n_ways=2)
        s.set_n_ways(5)
        assert len(s.output_ports) == 5


class TestPowerCombiner:
    def test_default_is_two_way_combiner(self):
        c = PowerCombiner()
        assert c.is_combiner is True
        assert len(c.input_ports) == 2
        assert len(c.output_ports) == 1

    def test_round_trip(self):
        c = PowerCombiner(n_ways=4)
        c2 = PowerCombiner.from_dict(c.to_dict())
        assert c2.n_ways == 4

    def test_combiner_preserves_distinct_input_frequencies(self):
        c = PowerCombiner(n_ways=3)
        s1 = make_signal(fc=1.0e9, pwr=0.0)
        s2 = make_signal(fc=1.2e9, pwr=0.0)
        s3 = make_signal(fc=1.4e9, pwr=0.0)
        c.process(s1, "IN0")
        c.process(s2, "IN1")
        out = c.process(s3, "IN2")["OUT"]
        freqs = sorted([out.carrier_frequency] + [s.frequency for s in out.spurs])
        assert freqs == pytest.approx([1.0e9, 1.2e9, 1.4e9])

    def test_combiner_output_total_power_scales_with_active_inputs(self):
        c = PowerCombiner(n_ways=3)
        s1 = make_signal(fc=1.0e9, pwr=0.0)
        s2 = make_signal(fc=1.2e9, pwr=0.0)
        c.process(s1, "IN0")
        out_two = c.process(s2, "IN1")["OUT"]
        c.reset_runtime_state()
        c.process(s1, "IN0")
        c.process(s2, "IN1")
        out_three = c.process(make_signal(fc=1.4e9, pwr=0.0), "IN2")["OUT"]
        delta_db = out_three.total_power_dbm() - out_two.total_power_dbm()
        assert delta_db == pytest.approx(10.0 * math.log10(3.0 / 2.0), abs=1e-6)

    def test_combiner_logs_constructive_sum_warning(self):
        c = PowerCombiner(n_ways=3)
        c.process(make_signal(fc=1.0e9, pwr=-3.0), "IN0")
        c.process(make_signal(fc=1.0e9, pwr=-3.0), "IN1")
        _ = c.process(make_signal(fc=2.0e9, pwr=-3.0), "IN2")
        msgs = c.pop_runtime_messages()
        assert any(level == "warning" and "constructive summation" in msg for level, msg in msgs)


# ======================================================================= #
# Switch                                                                   #
# ======================================================================= #

class TestSwitch:
    def test_1x2_has_correct_ports(self):
        sw = Switch(topology="1x2")
        assert len(sw.input_ports) == 1
        assert len(sw.output_ports) == 2

    def test_2x1_has_correct_ports(self):
        sw = Switch(topology="2x1")
        assert len(sw.input_ports) == 2
        assert len(sw.output_ports) == 1

    def test_1xn_has_correct_ports(self):
        sw = Switch(topology="1xN", n_ways=4)
        assert len(sw.input_ports) == 1
        assert len(sw.output_ports) == 4

    def test_nx1_has_correct_ports(self):
        sw = Switch(topology="Nx1", n_ways=5)
        assert len(sw.input_ports) == 5
        assert len(sw.output_ports) == 1

    def test_1x2_routes_to_active_port(self):
        sw = Switch(topology="1x2", active_port=0, insertion_loss_db=0.5, isolation_db=40.0)
        sig = make_signal(pwr=0.0)
        out = sw.process(sig)
        # OUT0 = active -> insertion_loss, OUT1 = isolated -> isolation
        assert out["OUT0"].power_dbm == pytest.approx(-0.5, abs=1e-6)
        assert out["OUT1"].power_dbm == pytest.approx(-40.0, abs=1e-6)

    def test_1x2_toggle_changes_routing(self):
        sw = Switch(topology="1x2", active_port=0, insertion_loss_db=0.5, isolation_db=40.0)
        sw.toggle_state()
        assert sw.active_port == 1
        sig = make_signal(pwr=0.0)
        out = sw.process(sig)
        assert out["OUT1"].power_dbm == pytest.approx(-0.5, abs=1e-6)
        assert out["OUT0"].power_dbm == pytest.approx(-40.0, abs=1e-6)

    def test_toggle_wraps_around(self):
        sw = Switch(topology="1x2", active_port=1)
        sw.toggle_state()
        assert sw.active_port == 0

    def test_toggle_wraps_around_for_arbitrary_n(self):
        sw = Switch(topology="1xN", n_ways=4, active_port=3)
        sw.toggle_state()
        assert sw.active_port == 0

    def test_round_trip(self):
        sw = Switch(topology="4x1", active_port=1, insertion_loss_db=1.0, isolation_db=35.0)
        assert sw.n_ways == 4
        sw2 = Switch.from_dict(sw.to_dict())
        assert sw2.topology == "Nx1"
        assert sw2.n_ways == 4
        assert sw2.active_port == 1
        assert sw2.insertion_loss_db == pytest.approx(1.0)
        assert sw2.isolation_db == pytest.approx(35.0)

    def test_gain_is_negative_insertion_loss(self):
        sw = Switch(insertion_loss_db=0.5)
        assert sw.gain_db == pytest.approx(-0.5)

    def test_default_switch_p1db_is_active(self):
        sw = Switch()
        assert sw.p1db_dbm is not None


# ======================================================================= #
# Source                                                                   #
# ======================================================================= #

class TestSource:
    def test_no_input_ports(self):
        s = Source()
        assert len(s.input_ports) == 0

    def test_one_output_port(self):
        s = Source()
        assert len(s.output_ports) == 1
        assert s.output_ports[0].name == "OUT"

    def test_generate_frequency_and_power(self):
        s = Source(frequency=2.4e9, output_power_dbm=-10.0, snr_db=55.0)
        sig = s.generate()
        assert sig.carrier_frequency == pytest.approx(2.4e9)
        assert sig.power_dbm == pytest.approx(-10.0)
        assert sig.snr_db == pytest.approx(55.0)

    def test_generate_has_no_spurs(self):
        s = Source(frequency=1e9, output_power_dbm=0.0)
        sig = s.generate()
        assert sig.spurs == []

    def test_process_returns_out(self):
        s = Source(frequency=1e9, output_power_dbm=5.0)
        out = s.process()
        assert "OUT" in out
        assert out["OUT"].power_dbm == pytest.approx(5.0)

    def test_round_trip(self):
        s = Source(frequency=5.8e9, output_power_dbm=-5.0, snr_db=44.0)
        s2 = Source.from_dict(s.to_dict())
        assert s2.frequency == pytest.approx(5.8e9)
        assert s2.output_power_dbm == pytest.approx(-5.0)
        assert s2.snr_db == pytest.approx(44.0)


# ======================================================================= #
# Sink                                                                     #
# ======================================================================= #

class TestSink:
    def test_no_output_ports(self):
        s = Sink()
        assert len(s.output_ports) == 0

    def test_one_input_port(self):
        s = Sink()
        assert len(s.input_ports) == 1

    def test_process_captures_signal(self):
        s = Sink()
        sig = make_signal(fc=2e9, pwr=-15.0)
        s.process(sig)
        assert s.last_signal is not None
        assert s.last_signal.power_dbm == pytest.approx(-15.0)
        assert s.last_signal.carrier_frequency == pytest.approx(2e9)

    def test_process_stores_deep_copy(self):
        """Mutating the original signal after process() must not change stored copy."""
        s = Sink()
        sig = make_signal(pwr=-15.0)
        s.process(sig)
        sig.power_dbm = 99.0
        assert s.last_signal.power_dbm == pytest.approx(-15.0)

    def test_process_returns_empty_dict(self):
        s = Sink()
        sig = make_signal()
        result = s.process(sig)
        assert result == {}

    def test_round_trip(self):
        s = Sink(label="Measurement Point")
        s2 = Sink.from_dict(s.to_dict())
        assert s2.label == "Measurement Point"
        assert s2.BLOCK_TYPE == "Sink"


# ======================================================================= #
# block_from_dict dispatcher                                               #
# ======================================================================= #

class TestBlockRegistry:
    @pytest.mark.parametrize("block_type,cls", [
        ("Amplifier",      Amplifier),
        ("Attenuator",     Attenuator),
        ("Mixer",          Mixer),
        ("SparBlock",      SparBlock),
        ("TransferFnBlock", TransferFnBlock),
        ("LowPassFilter",  LowPassFilter),
        ("HighPassFilter", HighPassFilter),
        ("PowerSplitter",  PowerSplitter),
        ("PowerCombiner",  PowerCombiner),
        ("Switch",         Switch),
        ("Source",         Source),
        ("Sink",           Sink),
    ])
    def test_registry_contains_type(self, block_type, cls):
        assert BLOCK_REGISTRY[block_type] is cls

    def test_block_from_dict_amplifier(self):
        a = Amplifier(gain_db=12.0)
        b = block_from_dict(a.to_dict())
        assert isinstance(b, Amplifier)
        assert b.gain_db == pytest.approx(12.0)

    def test_block_from_dict_sink(self):
        s = Sink(label="Out")
        b = block_from_dict(s.to_dict())
        assert isinstance(b, Sink)
        assert b.label == "Out"

    def test_block_from_dict_unknown_defaults_to_rfblock(self):
        d = {"block_type": "UnknownBlock", "label": "X", "gain_db": 5.0}
        b = block_from_dict(d)
        assert isinstance(b, RFBlock)

    def test_all_blocks_round_trip_via_registry(self):
        """Every registered block type must survive a to_dict -> block_from_dict round-trip."""
        blocks = [
            Amplifier(), Attenuator(), Mixer(), SparBlock(),
            TransferFnBlock(), LowPassFilter(), HighPassFilter(),
            PowerSplitter(), PowerCombiner(), Switch(), Source(), Sink(),
        ]
        for orig in blocks:
            restored = block_from_dict(orig.to_dict())
            assert type(restored) == type(orig), f"Type mismatch for {orig.BLOCK_TYPE}"
            assert restored.block_id == orig.block_id
            assert restored.label == orig.label


# ======================================================================= #
# Cross-block: signal propagation through a chain                         #
# ======================================================================= #

class TestSignalChainPropagation:
    def test_source_amp_sink_power(self):
        src = Source(frequency=1e9, output_power_dbm=-20.0)
        amp = Amplifier(gain_db=25.0)
        sink = Sink()

        sig = src.generate()
        sig = amp.process(sig)["OUT"]
        sink.process(sig)
        assert sink.last_signal.power_dbm == pytest.approx(5.0)

    def test_attenuator_after_amplifier(self):
        amp = Amplifier(gain_db=20.0)
        att = Attenuator(attenuation_db=10.0)
        sig = make_signal(pwr=-30.0)
        sig = amp.process(sig)["OUT"]
        sig = att.process(sig)["OUT"]
        assert sig.power_dbm == pytest.approx(-20.0)

    def test_lpf_removes_signal_above_cutoff(self):
        src = Source(frequency=10e9, output_power_dbm=0.0)
        filt = LowPassFilter(order=7, cutoff_hz=1e9)
        sig = src.generate()
        out = filt.process(sig)["OUT"]
        # 10x above cutoff, order 7: attenuation = 20*7*log10(10) = 140 dB
        assert out.power_dbm < -100.0

    def test_switch_routes_correctly_through_chain(self):
        """Signal routed through active switch port should have insertion loss only."""
        src = Source(frequency=1e9, output_power_dbm=0.0)
        sw = Switch(topology="1x2", active_port=0, insertion_loss_db=0.5, isolation_db=40.0)
        sig = src.generate()
        out = sw.process(sig)
        assert out["OUT0"].power_dbm == pytest.approx(-0.5, abs=1e-6)
        assert out["OUT1"].power_dbm == pytest.approx(-40.0, abs=1e-6)

    def test_mixer_down_converts_carrier(self):
        src = Source(frequency=2.45e9, output_power_dbm=0.0)
        lo = Source(frequency=2.4e9, output_power_dbm=0.0)
        mix = Mixer(conversion_expressions=["RF-LO"], gain_db=-7.0)
        sig = src.generate()
        mix.process(sig, "RF")
        out = mix.process(lo.generate(), "LO")["IF"]
        assert out.carrier_frequency == pytest.approx(50e6, rel=1e-6)
        assert out.power_dbm == pytest.approx(-7.0)

    def test_spur_accumulation_through_chain(self):
        """Spurs added by one block must survive subsequent gain stages."""
        amp1 = Amplifier(gain_db=20.0)
        amp1.spur_coefficients = [{"m": 3, "n": 0, "rel_power_db": -40.0}]
        amp2 = Amplifier(gain_db=10.0)
        # No spurs in amp2 definition

        sig = make_signal(fc=1e9, pwr=-30.0)
        sig = amp1.process(sig)["OUT"]
        # After amp1: carrier=-10dBm, spur at 3GHz: -10-40=-50dBm
        assert abs(sig.spurs[0].frequency - 3e9) < 1e6
        assert sig.spurs[0].power_dbm == pytest.approx(-50.0)

        sig = amp2.process(sig)["OUT"]
        # After amp2: carrier=0dBm, spur: -50+10=-40dBm
        assert sig.power_dbm == pytest.approx(0.0)
        assert sig.spurs[0].power_dbm == pytest.approx(-40.0)


# ======================================================================= #
# Hierarchical subcircuit simulation                                       #
# ======================================================================= #

class TestHierSubcircuitSimulation:
    def test_uses_internal_chain_not_pass_through(self, tmp_path):
        sub_path = tmp_path / "sub_amp.json"
        scene = {
            "version": "1",
            "metadata": {},
            "blocks": [
                {"block_type": "HierInputPin", "block_id": "in1", "pin_name": "IN", "label": "IN"},
                {"block_type": "Amplifier", "block_id": "amp1", "gain_db": 12.0, "nf_db": 2.0},
                {"block_type": "HierOutputPin", "block_id": "out1", "pin_name": "OUT", "label": "OUT"},
            ],
            "connections": [
                {"src_block_id": "in1", "src_port": "IN", "dst_block_id": "amp1", "dst_port": "IN"},
                {"src_block_id": "amp1", "src_port": "OUT", "dst_block_id": "out1", "dst_port": "OUT"},
            ],
            "annotations": [],
        }
        sub_path.write_text(json.dumps(scene), encoding="utf-8")

        block = HierSubcircuit(subcircuit_path=str(sub_path))
        out = block.process(make_signal(fc=2e9, pwr=-25.0), "IN")
        assert "OUT" in out
        assert out["OUT"].power_dbm == pytest.approx(-13.0, abs=1e-6)

    def test_multi_input_internal_mixer_requires_both_inputs(self, tmp_path):
        sub_path = tmp_path / "sub_mixer.json"
        scene = {
            "version": "1",
            "metadata": {},
            "blocks": [
                {"block_type": "HierInputPin", "block_id": "in_rf", "pin_name": "RF", "label": "RF"},
                {"block_type": "HierInputPin", "block_id": "in_lo", "pin_name": "LO", "label": "LO"},
                {"block_type": "Mixer", "block_id": "mix1", "conversion_expressions": ["RF-LO"], "gain_db": -7.0},
                {"block_type": "HierOutputPin", "block_id": "out_if", "pin_name": "IF", "label": "IF"},
            ],
            "connections": [
                {"src_block_id": "in_rf", "src_port": "RF", "dst_block_id": "mix1", "dst_port": "RF"},
                {"src_block_id": "in_lo", "src_port": "LO", "dst_block_id": "mix1", "dst_port": "LO"},
                {"src_block_id": "mix1", "src_port": "IF", "dst_block_id": "out_if", "dst_port": "IF"},
            ],
            "annotations": [],
        }
        sub_path.write_text(json.dumps(scene), encoding="utf-8")

        block = HierSubcircuit(subcircuit_path=str(sub_path))
        assert block.process(make_signal(fc=2e9, pwr=-10.0), "RF") == {}

        out = block.process(make_signal(fc=1.9e9, pwr=0.0), "LO")
        assert "IF" in out
        assert out["IF"].carrier_frequency == pytest.approx(100e6, rel=1e-9)
        assert out["IF"].power_dbm == pytest.approx(-17.0, abs=1e-6)

    def test_combiner_preserves_combined_noise_floor(self, tmp_path):
        sub_path = tmp_path / "sub_combiner_nf.json"
        scene = {
            "version": "1",
            "metadata": {},
            "blocks": [
                {"block_type": "HierInputPin", "block_id": "in0", "pin_name": "IN0", "label": "IN0"},
                {"block_type": "HierInputPin", "block_id": "in1", "pin_name": "IN1", "label": "IN1"},
                {"block_type": "PowerCombiner", "block_id": "cmb", "n_ways": 2, "label": "2-way Combiner"},
                {"block_type": "HierOutputPin", "block_id": "out1", "pin_name": "OUT", "label": "OUT"},
            ],
            "connections": [
                {"src_block_id": "in0", "src_port": "IN0", "dst_block_id": "cmb", "dst_port": "IN0"},
                {"src_block_id": "in1", "src_port": "IN1", "dst_block_id": "cmb", "dst_port": "IN1"},
                {"src_block_id": "cmb", "src_port": "OUT", "dst_block_id": "out1", "dst_port": "OUT"},
            ],
            "annotations": [],
        }
        sub_path.write_text(json.dumps(scene), encoding="utf-8")

        block = HierSubcircuit(subcircuit_path=str(sub_path))
        s0 = make_signal(fc=1.0e9, pwr=0.0)
        s1 = make_signal(fc=2.0e9, pwr=0.0)
        s0.set_noise_floor_dbm(-100.0)
        s1.set_noise_floor_dbm(-100.0)

        _ = block.process(s0, "IN0")
        out = block.process(s1, "IN1")
        assert "OUT" in out
        assert out["OUT"].get_noise_floor_dbm() == pytest.approx(-100.0, abs=1e-6)


class TestHierSubcircuitAnalysisFlattening:
    def test_flattens_internal_chain_for_analysis(self, tmp_path):
        sub_path = tmp_path / "sub_chain.json"
        scene = {
            "version": "1",
            "metadata": {},
            "blocks": [
                {"block_type": "HierInputPin", "block_id": "in1", "pin_name": "IN", "label": "IN"},
                {"block_type": "Amplifier", "block_id": "amp1", "gain_db": 12.0, "nf_db": 2.0, "label": "A1"},
                {"block_type": "Attenuator", "block_id": "att1", "attenuation_db": 3.0, "label": "AT1"},
                {"block_type": "HierOutputPin", "block_id": "out1", "pin_name": "OUT", "label": "OUT"},
            ],
            "connections": [
                {"src_block_id": "in1", "src_port": "IN", "dst_block_id": "amp1", "dst_port": "IN"},
                {"src_block_id": "amp1", "src_port": "OUT", "dst_block_id": "att1", "dst_port": "IN"},
                {"src_block_id": "att1", "src_port": "OUT", "dst_block_id": "out1", "dst_port": "OUT"},
            ],
            "annotations": [],
        }
        sub_path.write_text(json.dumps(scene), encoding="utf-8")

        blocks = analysis_blocks_from_subcircuit(str(sub_path))
        assert [b.BLOCK_TYPE for b in blocks] == ["Amplifier", "Attenuator"]
        assert [b.label for b in blocks] == ["A1", "AT1"]

    def test_flattens_nested_subcircuits_for_analysis(self, tmp_path):
        child_path = tmp_path / "child.json"
        child_scene = {
            "version": "1",
            "metadata": {},
            "blocks": [
                {"block_type": "HierInputPin", "block_id": "cin", "pin_name": "IN", "label": "IN"},
                {"block_type": "Amplifier", "block_id": "camp", "gain_db": 10.0, "label": "ChildAmp"},
                {"block_type": "HierOutputPin", "block_id": "cout", "pin_name": "OUT", "label": "OUT"},
            ],
            "connections": [
                {"src_block_id": "cin", "src_port": "IN", "dst_block_id": "camp", "dst_port": "IN"},
                {"src_block_id": "camp", "src_port": "OUT", "dst_block_id": "cout", "dst_port": "OUT"},
            ],
            "annotations": [],
        }
        child_path.write_text(json.dumps(child_scene), encoding="utf-8")

        parent_path = tmp_path / "parent.json"
        parent_scene = {
            "version": "1",
            "metadata": {},
            "blocks": [
                {"block_type": "HierInputPin", "block_id": "pin", "pin_name": "IN", "label": "IN"},
                {"block_type": "HierSubcircuit", "block_id": "sub1", "subcircuit_path": "child.json", "label": "Sub"},
                {"block_type": "Attenuator", "block_id": "patt", "attenuation_db": 2.0, "label": "ParentAtt"},
                {"block_type": "HierOutputPin", "block_id": "pout", "pin_name": "OUT", "label": "OUT"},
            ],
            "connections": [
                {"src_block_id": "pin", "src_port": "IN", "dst_block_id": "sub1", "dst_port": "IN"},
                {"src_block_id": "sub1", "src_port": "OUT", "dst_block_id": "patt", "dst_port": "IN"},
                {"src_block_id": "patt", "src_port": "OUT", "dst_block_id": "pout", "dst_port": "OUT"},
            ],
            "annotations": [],
        }
        parent_path.write_text(json.dumps(parent_scene), encoding="utf-8")

        blocks = analysis_blocks_from_subcircuit(str(parent_path))
        assert [b.BLOCK_TYPE for b in blocks] == ["Amplifier", "Attenuator"]
        assert [b.label for b in blocks] == ["ChildAmp", "ParentAtt"]
