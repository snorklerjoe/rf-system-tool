"""
Unit tests for the Signal data model.

Tests cover:
  - Construction and default values
  - apply_gain: carrier and spur power scaling
  - add_spur: add new, update existing
  - copy: deep-copy isolation
  - SpurTone helpers
  - Serialisation round-trip (to_dict / from_dict)
  - Edge cases (zero gain, negative gain, empty spurs)
"""
import math
import pytest
from rf_tool.models.signal import Signal, SpurTone


# ======================================================================= #
# SpurTone                                                                 #
# ======================================================================= #

class TestSpurTone:
    def test_construction(self):
        s = SpurTone(frequency=2.4e9, power_dbm=-30.0)
        assert s.frequency == pytest.approx(2.4e9)
        assert s.power_dbm == pytest.approx(-30.0)

    def test_scale_power_positive(self):
        s = SpurTone(1e9, -20.0)
        scaled = s.scale_power(10.0)
        assert scaled.frequency == pytest.approx(1e9)
        assert scaled.power_dbm == pytest.approx(-10.0)

    def test_scale_power_negative(self):
        s = SpurTone(1e9, -20.0)
        scaled = s.scale_power(-5.0)
        assert scaled.power_dbm == pytest.approx(-25.0)

    def test_scale_power_zero(self):
        s = SpurTone(1e9, -20.0)
        scaled = s.scale_power(0.0)
        assert scaled.power_dbm == pytest.approx(-20.0)

    def test_scale_does_not_mutate_original(self):
        s = SpurTone(1e9, -20.0)
        _ = s.scale_power(10.0)
        assert s.power_dbm == pytest.approx(-20.0)

    def test_serialisation_round_trip(self):
        s = SpurTone(2.5e9, -45.5)
        d = s.to_dict()
        s2 = SpurTone.from_dict(d)
        assert s2.frequency == pytest.approx(s.frequency)
        assert s2.power_dbm == pytest.approx(s.power_dbm)


# ======================================================================= #
# Signal construction                                                      #
# ======================================================================= #

class TestSignalConstruction:
    def test_defaults(self):
        sig = Signal(carrier_frequency=1e9, power_dbm=-10.0)
        assert sig.carrier_frequency == pytest.approx(1e9)
        assert sig.power_dbm == pytest.approx(-10.0)
        assert sig.spurs == []
        assert sig.phase_noise_dbc_hz == {}
        assert sig.snr_db is None

    def test_with_spurs(self):
        spur = SpurTone(2e9, -50.0)
        sig = Signal(1e9, 0.0, spurs=[spur])
        assert len(sig.spurs) == 1
        assert sig.spurs[0].frequency == pytest.approx(2e9)

    def test_with_phase_noise(self):
        pn = {1e3: -80.0, 10e3: -100.0}
        sig = Signal(1e9, 0.0, phase_noise_dbc_hz=pn)
        assert sig.phase_noise_dbc_hz[1e3] == pytest.approx(-80.0)
        assert sig.phase_noise_dbc_hz[10e3] == pytest.approx(-100.0)


# ======================================================================= #
# Signal.copy                                                              #
# ======================================================================= #

class TestSignalCopy:
    def test_copy_is_equal(self):
        spur = SpurTone(2e9, -40.0)
        sig = Signal(1e9, -10.0, spurs=[spur])
        c = sig.copy()
        assert c.carrier_frequency == pytest.approx(sig.carrier_frequency)
        assert c.power_dbm == pytest.approx(sig.power_dbm)
        assert len(c.spurs) == len(sig.spurs)
        assert c.spurs[0].frequency == pytest.approx(2e9)
        assert c.spurs[0].power_dbm == pytest.approx(-40.0)

    def test_copy_is_independent(self):
        """Mutating the copy must not affect the original."""
        spur = SpurTone(2e9, -40.0)
        sig = Signal(1e9, -10.0, spurs=[spur])
        c = sig.copy()
        c.power_dbm = 99.0
        c.spurs[0].power_dbm = 0.0
        # Original unchanged
        assert sig.power_dbm == pytest.approx(-10.0)
        assert sig.spurs[0].power_dbm == pytest.approx(-40.0)

    def test_copy_phase_noise_independent(self):
        pn = {1e3: -80.0}
        sig = Signal(1e9, 0.0, phase_noise_dbc_hz=pn)
        c = sig.copy()
        c.phase_noise_dbc_hz[1e3] = 0.0
        assert sig.phase_noise_dbc_hz[1e3] == pytest.approx(-80.0)


# ======================================================================= #
# Signal.apply_gain                                                        #
# ======================================================================= #

class TestApplyGain:
    def test_carrier_gain_positive(self):
        sig = Signal(1e9, -10.0)
        out = sig.apply_gain(20.0)
        assert out.power_dbm == pytest.approx(10.0)

    def test_carrier_gain_negative(self):
        sig = Signal(1e9, 0.0)
        out = sig.apply_gain(-3.0)
        assert out.power_dbm == pytest.approx(-3.0)

    def test_carrier_gain_zero(self):
        sig = Signal(1e9, -5.0)
        out = sig.apply_gain(0.0)
        assert out.power_dbm == pytest.approx(-5.0)

    def test_spur_gain_applied(self):
        spur = SpurTone(2e9, -30.0)
        sig = Signal(1e9, 0.0, spurs=[spur])
        out = sig.apply_gain(10.0)
        assert out.spurs[0].power_dbm == pytest.approx(-20.0)

    def test_multiple_spurs_all_scaled(self):
        spurs = [SpurTone(f * 1e9, -40.0 - i * 5) for i, f in enumerate([2, 3, 4])]
        sig = Signal(1e9, 0.0, spurs=spurs)
        out = sig.apply_gain(15.0)
        for i, s in enumerate(out.spurs):
            expected = -40.0 - i * 5 + 15.0
            assert s.power_dbm == pytest.approx(expected), f"Spur {i} mismatch"

    def test_apply_gain_does_not_mutate_original(self):
        spur = SpurTone(2e9, -30.0)
        sig = Signal(1e9, 0.0, spurs=[spur])
        _ = sig.apply_gain(10.0)
        assert sig.power_dbm == pytest.approx(0.0)
        assert sig.spurs[0].power_dbm == pytest.approx(-30.0)

    def test_frequency_unchanged_by_gain(self):
        sig = Signal(2.4e9, -20.0)
        out = sig.apply_gain(30.0)
        assert out.carrier_frequency == pytest.approx(2.4e9)

    def test_chained_gains(self):
        sig = Signal(1e9, -20.0)
        out = sig.apply_gain(10.0).apply_gain(5.0).apply_gain(-3.0)
        assert out.power_dbm == pytest.approx(-8.0)


# ======================================================================= #
# Signal.add_spur                                                          #
# ======================================================================= #

class TestAddSpur:
    def test_add_new_spur(self):
        sig = Signal(1e9, 0.0)
        sig.add_spur(2e9, -50.0)
        assert len(sig.spurs) == 1
        assert sig.spurs[0].frequency == pytest.approx(2e9)
        assert sig.spurs[0].power_dbm == pytest.approx(-50.0)

    def test_add_multiple_distinct_spurs(self):
        sig = Signal(1e9, 0.0)
        sig.add_spur(2e9, -50.0)
        sig.add_spur(3e9, -60.0)
        assert len(sig.spurs) == 2

    def test_update_existing_spur_same_frequency(self):
        """Adding a spur at an existing frequency should update, not duplicate."""
        sig = Signal(1e9, 0.0)
        sig.add_spur(2e9, -50.0)
        sig.add_spur(2e9, -30.0)   # update
        assert len(sig.spurs) == 1
        assert sig.spurs[0].power_dbm == pytest.approx(-30.0)

    def test_frequency_tolerance_update(self):
        """Frequencies within 1 mHz of each other should be treated as the same tone."""
        sig = Signal(1e9, 0.0)
        sig.add_spur(2e9, -50.0)
        sig.add_spur(2e9 + 0.0009, -35.0)  # within 1 mHz
        assert len(sig.spurs) == 1
        assert sig.spurs[0].power_dbm == pytest.approx(-35.0)

    def test_frequency_outside_tolerance_adds_new(self):
        sig = Signal(1e9, 0.0)
        sig.add_spur(2e9, -50.0)
        sig.add_spur(2e9 + 0.002, -35.0)  # outside 1 mHz tolerance
        assert len(sig.spurs) == 2


# ======================================================================= #
# Signal serialisation                                                     #
# ======================================================================= #

class TestSignalSerialisation:
    def test_to_dict_keys(self):
        sig = Signal(1e9, -10.0)
        d = sig.to_dict()
        for key in ("carrier_frequency", "power_dbm", "spurs", "phase_noise_dbc_hz", "snr_db", "noise_floor_dbm"):
            assert key in d

    def test_noise_floor_helpers_round_trip(self):
        sig = Signal(1e9, -10.0, snr_db=40.0)
        assert sig.get_noise_floor_dbm() == pytest.approx(-50.0)
        sig.set_noise_floor_dbm(-60.0)
        assert sig.noise_floor_dbm == pytest.approx(-60.0)
        assert sig.snr_db == pytest.approx(50.0)

    def test_round_trip_with_snr(self):
        sig = Signal(1e9, -12.0, snr_db=48.0)
        rt = Signal.from_dict(sig.to_dict())
        assert rt.snr_db == pytest.approx(48.0)

    def test_round_trip_no_spurs(self):
        sig = Signal(2.4e9, -15.0)
        d = sig.to_dict()
        sig2 = Signal.from_dict(d)
        assert sig2.carrier_frequency == pytest.approx(sig.carrier_frequency)
        assert sig2.power_dbm == pytest.approx(sig.power_dbm)
        assert sig2.spurs == []

    def test_round_trip_with_spurs(self):
        spurs = [SpurTone(2e9, -40.0), SpurTone(3e9, -55.0)]
        sig = Signal(1e9, 0.0, spurs=spurs)
        sig2 = Signal.from_dict(sig.to_dict())
        assert len(sig2.spurs) == 2
        assert sig2.spurs[0].frequency == pytest.approx(2e9)
        assert sig2.spurs[1].power_dbm == pytest.approx(-55.0)

    def test_round_trip_with_phase_noise(self):
        pn = {1e3: -80.0, 1e4: -100.0, 1e6: -150.0}
        sig = Signal(1e9, 0.0, phase_noise_dbc_hz=pn)
        sig2 = Signal.from_dict(sig.to_dict())
        for offset, val in pn.items():
            assert sig2.phase_noise_dbc_hz[offset] == pytest.approx(val)

    def test_repr_contains_frequency_and_power(self):
        sig = Signal(1e9, -10.0)
        r = repr(sig)
        assert "GHz" in r
        assert "dBm" in r
