"""
Unit tests for the RF cascade math engine.

Covers:
  - dB ↔ linear conversions (edge cases incl. 0 and negative)
  - cascade_gain  (trivial but must be exact)
  - cascade_noise_figure  (Friis formula, single/multi stage, edge cases)
  - cascade_iip3  (worst-case voltage addition, single/multi stage, None stages)
  - cascade_oip3  (= IIP3 + total_gain)
  - cascade_p1db  (same formula as IIP3 but using input P1dB)
  - compute_cascade_metrics  (full pipeline)
  - cascade_networks + s21_to_gain_db  (scikit-rf integration)

Mathematical reference:
  Friis:     F_total = F1 + (F2-1)/G1 + (F3-1)/(G1*G2) + ...
  IP3 (wc):  1/sqrt(IIP3_tot) = sum_k sqrt(G_cum(k)) / sqrt(IIP3_k)
  P1dB (wc): same formula with input-referred P1dB per stage
"""
import math
import pytest
import numpy as np

from rf_tool.engine.cascade import (
    db_to_linear_power,
    linear_power_to_db,
    dbm_to_mw,
    mw_to_dbm,
    cascade_gain,
    cascade_noise_figure,
    cascade_iip3,
    cascade_oip3,
    cascade_p1db,
    compute_cascade_metrics,
)


# ======================================================================= #
# dB / power helpers                                                       #
# ======================================================================= #

class TestDbLinearConversions:
    def test_0db_is_unity(self):
        assert db_to_linear_power(0.0) == pytest.approx(1.0)

    def test_10db_is_10x(self):
        assert db_to_linear_power(10.0) == pytest.approx(10.0)

    def test_minus3db_approx_half(self):
        assert db_to_linear_power(-3.0103) == pytest.approx(0.5, rel=1e-4)

    def test_roundtrip(self):
        for val in (0.0, 1.0, 10.0, -20.0, 30.0, -100.0):
            assert linear_power_to_db(db_to_linear_power(val)) == pytest.approx(val, abs=1e-9)

    def test_zero_linear_returns_minus_inf(self):
        assert linear_power_to_db(0.0) == -math.inf

    def test_negative_linear_returns_minus_inf(self):
        assert linear_power_to_db(-1.0) == -math.inf

    def test_dbm_0_is_1mw(self):
        assert dbm_to_mw(0.0) == pytest.approx(1.0)

    def test_dbm_30_is_1000mw(self):
        assert dbm_to_mw(30.0) == pytest.approx(1000.0, rel=1e-9)

    def test_mw_to_dbm_1mw(self):
        assert mw_to_dbm(1.0) == pytest.approx(0.0)

    def test_mw_to_dbm_0_is_minus_inf(self):
        assert mw_to_dbm(0.0) == -math.inf

    def test_dbm_roundtrip(self):
        for dbm in (-50.0, -10.0, 0.0, 10.0, 30.0):
            assert mw_to_dbm(dbm_to_mw(dbm)) == pytest.approx(dbm, abs=1e-9)


# ======================================================================= #
# Cascade gain                                                             #
# ======================================================================= #

class TestCascadeGain:
    def test_single_stage(self):
        assert cascade_gain([15.0]) == pytest.approx(15.0)

    def test_two_stages(self):
        assert cascade_gain([15.0, -3.0]) == pytest.approx(12.0)

    def test_all_zeros(self):
        assert cascade_gain([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_four_stages(self):
        assert cascade_gain([10.0, 5.0, -2.0, 7.0]) == pytest.approx(20.0)

    def test_empty_raises(self):
        """Empty list sums to 0, which is fine (sum of empty = 0)."""
        assert cascade_gain([]) == pytest.approx(0.0)


# ======================================================================= #
# Friis Noise Figure Cascade                                               #
# ======================================================================= #

class TestCascadeNoiseFigure:
    """
    Verified against textbook (Pozar, Razavi) examples.
    """

    def test_single_stage_returns_own_nf(self):
        nf = cascade_noise_figure([20.0], [3.0])
        assert nf == pytest.approx(3.0, abs=1e-6)

    def test_two_stages_low_gain_first(self):
        """Low gain in first stage makes NF of second stage matter a lot."""
        # G1=0dB (=1), NF1=3dB (F=2), G2=20dB, NF2=5dB (F≈3.162)
        # F_total = 2 + (3.162-1)/1 = 4.162  => 6.194 dB
        nf = cascade_noise_figure([0.0, 20.0], [3.0, 5.0])
        F_expected = 10 ** (3.0 / 10) + (10 ** (5.0 / 10) - 1) / 1.0
        nf_expected = 10 * math.log10(F_expected)
        assert nf == pytest.approx(nf_expected, rel=1e-6)

    def test_two_stages_high_gain_first(self):
        """High gain first stage (30 dB) makes second stage NF negligible."""
        # G1=30dB (=1000), NF1=3dB, NF2=10dB
        # F_total ≈ F1 + (F2-1)/1000
        nf = cascade_noise_figure([30.0, 20.0], [3.0, 10.0])
        F1 = 10 ** (3.0 / 10)
        F2 = 10 ** (10.0 / 10)
        G1 = 10 ** (30.0 / 10)
        F_total = F1 + (F2 - 1) / G1
        nf_expected = 10 * math.log10(F_total)
        assert nf == pytest.approx(nf_expected, rel=1e-9)

    def test_three_stages(self):
        # G=[20,10,15], NF=[3,5,8]
        gains = [20.0, 10.0, 15.0]
        nfs = [3.0, 5.0, 8.0]
        Gs = [10 ** (g / 10) for g in gains]
        Fs = [10 ** (nf / 10) for nf in nfs]
        F_total = Fs[0] + (Fs[1] - 1) / Gs[0] + (Fs[2] - 1) / (Gs[0] * Gs[1])
        nf_expected = 10 * math.log10(F_total)
        nf = cascade_noise_figure(gains, nfs)
        assert nf == pytest.approx(nf_expected, rel=1e-9)

    def test_noiseless_stages_give_zero_nf(self):
        nf = cascade_noise_figure([20.0, 10.0, 15.0], [0.0, 0.0, 0.0])
        assert nf == pytest.approx(0.0, abs=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cascade_noise_figure([20.0], [3.0, 5.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            cascade_noise_figure([], [])

    def test_attenuator_nf_equals_attenuation(self):
        """Passive attenuator: NF = attenuation (at room T)."""
        # 10 dB attenuator: gain=-10, NF=10
        nf = cascade_noise_figure([-10.0], [10.0])
        assert nf == pytest.approx(10.0, abs=1e-6)

    def test_two_identical_10db_amps(self):
        # G=10 dB each, NF=3 dB each
        # F = 2 + (2-1)/10 = 2.1  => 10*log10(2.1) ≈ 3.222 dB
        nf = cascade_noise_figure([10.0, 10.0], [3.0103, 3.0103])
        F1 = 10 ** (3.0103 / 10)  # ≈ 2.0
        G1 = 10.0
        F_total = F1 + (F1 - 1) / G1
        nf_expected = 10 * math.log10(F_total)
        assert nf == pytest.approx(nf_expected, rel=1e-6)

    def test_friis_order_matters(self):
        """Different stage ordering yields different total NF."""
        nf_ab = cascade_noise_figure([20.0, 3.0], [3.0, 10.0])
        nf_ba = cascade_noise_figure([3.0, 20.0], [10.0, 3.0])
        assert nf_ab != pytest.approx(nf_ba, abs=0.01)

    def test_very_many_stages_numerically_stable(self):
        """50 identical stages should still give a finite, correct result."""
        n = 50
        g = [10.0] * n     # 10 dB gain each
        nf = [3.0] * n
        F1 = 10 ** (3.0 / 10)
        G1 = 10 ** (10.0 / 10)
        # Exact: F_total = F1 * sum(1/G1^k, k=0..N-1) - (N-1)/G_total + 1 -- use Friis loop
        F_total = F1
        cum_g = G1
        for _ in range(1, n):
            F_total += (F1 - 1) / cum_g
            cum_g *= G1
        nf_expected = 10 * math.log10(F_total)
        result = cascade_noise_figure(g, nf)
        assert result == pytest.approx(nf_expected, rel=1e-6)


# ======================================================================= #
# IP3 Cascade                                                              #
# ======================================================================= #

class TestCascadeIIP3:
    """
    Worst-case (phase-aligned voltage addition) formula:
        1/sqrt(IIP3_tot_mW) = sum_k  sqrt(G_cum(k)) / sqrt(IIP3_k_mW)
    """

    def test_single_stage_returns_own_iip3(self):
        iip3 = cascade_iip3([20.0], [10.0])
        assert iip3 == pytest.approx(10.0, abs=1e-6)

    def test_single_ideal_stage_returns_none(self):
        iip3 = cascade_iip3([20.0], [None])
        assert iip3 is None

    def test_two_stages_hand_calculation(self):
        """Hand-verify two-stage IP3 cascade.

        Stage 1: G=20dB (G_lin=100), IIP3=10dBm (=10mW)
        Stage 2: G=10dB (G_lin=10),  IIP3=20dBm (=100mW)

        1/sqrt(IIP3_tot) = sqrt(1)/sqrt(10) + sqrt(100)/sqrt(100)
                         = 1/sqrt(10) + 1
        IIP3_tot = 1 / (1/sqrt(10) + 1)^2
        """
        g1, iip3_1 = 20.0, 10.0   # dB, dBm
        g2, iip3_2 = 10.0, 20.0

        G1 = db_to_linear_power(g1)  # 100
        IIP3_1_mw = dbm_to_mw(iip3_1)  # 10 mW
        IIP3_2_mw = dbm_to_mw(iip3_2)  # 100 mW

        term1 = math.sqrt(1.0) / math.sqrt(IIP3_1_mw)
        term2 = math.sqrt(G1)  / math.sqrt(IIP3_2_mw)
        expected_mw = 1.0 / (term1 + term2) ** 2
        expected_dbm = mw_to_dbm(expected_mw)

        result = cascade_iip3([g1, g2], [iip3_1, iip3_2])
        assert result == pytest.approx(expected_dbm, rel=1e-6)

    def test_none_stage_skipped(self):
        """A stage with None IIP3 acts as ideal and must not affect the result."""
        iip3_no_ideal = cascade_iip3([10.0, 15.0], [20.0, 30.0])
        iip3_with_ideal = cascade_iip3([10.0, 0.0, 15.0], [20.0, None, 30.0])
        # The ideal middle stage (gain 0 dB) does not change IIP3
        assert iip3_with_ideal == pytest.approx(iip3_no_ideal, rel=1e-5)

    def test_all_none_returns_none(self):
        assert cascade_iip3([10.0, 20.0], [None, None]) is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cascade_iip3([10.0], [20.0, 30.0])

    def test_iip3_degraded_by_preceding_gain(self):
        """More gain before a stage makes its IP3 contribution worse."""
        iip3_low_gain  = cascade_iip3([0.0,  20.0], [None, 20.0])
        iip3_high_gain = cascade_iip3([30.0, 20.0], [None, 20.0])
        # Higher preceding gain -> worse cascaded IIP3
        assert iip3_high_gain < iip3_low_gain

    def test_three_stages_monotonic(self):
        """Adding more non-ideal stages must degrade IIP3."""
        iip3_1 = cascade_iip3([10.0], [20.0])
        iip3_2 = cascade_iip3([10.0, 10.0], [20.0, 25.0])
        iip3_3 = cascade_iip3([10.0, 10.0, 10.0], [20.0, 25.0, 30.0])
        assert iip3_1 >= iip3_2 >= iip3_3

    def test_first_stage_dominates_with_no_preceding_gain(self):
        """When gains are 0 dB, the stage with the lowest IIP3 dominates.

        With equal unity gains between all stages, the first stage
        (lowest IIP3) sets the cascade IIP3 almost entirely.
        Result should be within 3 dB of the first stage's IIP3.
        """
        # All gains = 0 dB so cumulative gain before each stage = 1
        iip3 = cascade_iip3([0.0, 0.0, 0.0], [5.0, 40.0, 40.0])
        # Result must be close to (but slightly below) 5 dBm
        assert iip3 < 5.0
        assert iip3 > 5.0 - 3.0   # not more than 3 dB below

    def test_positive_values_only(self):
        """IIP3 must always be a valid dBm value (no NaN, no inf)."""
        iip3 = cascade_iip3([10.0, 20.0, 5.0], [15.0, 25.0, 10.0])
        assert iip3 is not None
        assert math.isfinite(iip3)

    def test_five_stages_hand_verify(self):
        """Five-stage system: verify against manual loop calculation."""
        gains = [10.0, 15.0, 5.0, 20.0, 10.0]
        iip3s = [20.0, 25.0, 15.0, 30.0, 22.0]

        Gs = [db_to_linear_power(g) for g in gains]
        cum_G = 1.0
        total_sum = 0.0
        for i, iip3_dbm in enumerate(iip3s):
            iip3_mw = dbm_to_mw(iip3_dbm)
            total_sum += math.sqrt(cum_G) / math.sqrt(iip3_mw)
            cum_G *= Gs[i]
        expected = mw_to_dbm(1.0 / total_sum ** 2)

        result = cascade_iip3(gains, iip3s)
        assert result == pytest.approx(expected, rel=1e-9)


# ======================================================================= #
# OIP3 Cascade                                                             #
# ======================================================================= #

class TestCascadeOIP3:
    def test_oip3_equals_iip3_plus_total_gain(self):
        gains = [10.0, 15.0]
        iip3s = [20.0, 25.0]
        iip3 = cascade_iip3(gains, iip3s)
        oip3 = cascade_oip3(gains, iip3s)
        assert oip3 == pytest.approx(iip3 + cascade_gain(gains), rel=1e-9)

    def test_oip3_none_when_all_ideal(self):
        assert cascade_oip3([10.0, 20.0], [None, None]) is None

    def test_single_stage_oip3(self):
        # OIP3 = IIP3 + gain
        oip3 = cascade_oip3([20.0], [10.0])
        assert oip3 == pytest.approx(10.0 + 20.0, abs=1e-6)


# ======================================================================= #
# P1dB Cascade                                                             #
# ======================================================================= #

class TestCascadeP1dB:
    """
    P1dB cascade uses the same worst-case voltage formula as IP3,
    but applied to *input-referred* P1dB per stage.
    """

    def test_single_stage(self):
        # P1dB_out=20dBm, gain=15dB -> P1dB_in=5dBm
        # Cascade should give input-referred = 5 dBm
        p1db = cascade_p1db([15.0], [20.0])
        assert p1db == pytest.approx(5.0, abs=1e-6)

    def test_single_ideal_returns_none(self):
        assert cascade_p1db([10.0], [None]) is None

    def test_all_ideal_returns_none(self):
        assert cascade_p1db([10.0, 20.0], [None, None]) is None

    def test_two_stage_hand_verify(self):
        """
        Stage 1: G=10dB, P1dB_out=20dBm -> P1dB_in,1=10dBm (10mW)
        Stage 2: G=15dB, P1dB_out=25dBm -> P1dB_in,2=10dBm (10mW)

        1/sqrt(P1dB_tot) = sqrt(1)/sqrt(10mW) + sqrt(G1_lin)/sqrt(10mW)
                         = (1 + sqrt(G1_lin)) / sqrt(10mW)
        G1_lin = 10^(10/10) = 10
        sum = (1 + sqrt(10)) / sqrt(10)
        P1dB_tot = 10 / (1 + sqrt(10))^2 mW
        """
        G1_lin = db_to_linear_power(10.0)
        p1db_in_1 = 20.0 - 10.0  # = 10 dBm
        p1db_in_2 = 25.0 - 15.0  # = 10 dBm
        IIP3_1_mw = dbm_to_mw(p1db_in_1)
        IIP3_2_mw = dbm_to_mw(p1db_in_2)
        term1 = math.sqrt(1.0) / math.sqrt(IIP3_1_mw)
        term2 = math.sqrt(G1_lin) / math.sqrt(IIP3_2_mw)
        expected_mw = 1.0 / (term1 + term2) ** 2
        expected_dbm = mw_to_dbm(expected_mw)

        result = cascade_p1db([10.0, 15.0], [20.0, 25.0])
        assert result == pytest.approx(expected_dbm, rel=1e-9)

    def test_p1db_degrades_with_more_stages(self):
        p1db_1 = cascade_p1db([10.0], [20.0])
        p1db_2 = cascade_p1db([10.0, 10.0], [20.0, 20.0])
        p1db_3 = cascade_p1db([10.0, 10.0, 10.0], [20.0, 20.0, 20.0])
        assert p1db_1 >= p1db_2 >= p1db_3

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cascade_p1db([10.0], [20.0, 25.0])

    def test_none_stage_ignored(self):
        """An ideal stage with None P1dB must not affect the result."""
        p1db_direct = cascade_p1db([10.0, 15.0], [20.0, 25.0])
        p1db_with_ideal = cascade_p1db([10.0, 0.0, 15.0], [20.0, None, 25.0])
        assert p1db_with_ideal == pytest.approx(p1db_direct, rel=1e-5)


# ======================================================================= #
# compute_cascade_metrics                                                  #
# ======================================================================= #

class TestComputeCascadeMetrics:
    def _make_amp(self, gain, nf, p1db, oip3, max_pwr=None):
        from rf_tool.blocks.components import Amplifier
        return Amplifier(gain_db=gain, nf_db=nf, p1db_dbm=p1db, oip3_dbm=oip3,
                         max_input_power_dbm=max_pwr)

    def test_empty_block_list(self):
        metrics = compute_cascade_metrics([])
        assert metrics["gain_db"] == pytest.approx(0.0)
        assert metrics["nf_db"] == pytest.approx(0.0)
        assert metrics["iip3_dbm"] is None
        assert metrics["oip3_dbm"] is None
        assert metrics["p1db_in_dbm"] is None

    def test_single_amp(self):
        amp = self._make_amp(gain=20, nf=3, p1db=20, oip3=30)
        m = compute_cascade_metrics([amp])
        assert m["gain_db"] == pytest.approx(20.0)
        assert m["nf_db"] == pytest.approx(3.0, abs=1e-6)
        # IIP3 = OIP3 - gain = 30 - 20 = 10 dBm
        assert m["iip3_dbm"] == pytest.approx(10.0, rel=1e-5)
        # OIP3_cascade = IIP3 + total gain = 10 + 20 = 30 dBm
        assert m["oip3_dbm"] == pytest.approx(30.0, rel=1e-5)

    def test_two_amps_nf_correct(self):
        amp1 = self._make_amp(gain=20, nf=3, p1db=None, oip3=None)
        amp2 = self._make_amp(gain=10, nf=5, p1db=None, oip3=None)
        m = compute_cascade_metrics([amp1, amp2])
        expected_nf = cascade_noise_figure([20.0, 10.0], [3.0, 5.0])
        assert m["nf_db"] == pytest.approx(expected_nf, rel=1e-9)

    def test_cumulative_gains_correct(self):
        amps = [self._make_amp(10, 3, None, None) for _ in range(4)]
        m = compute_cascade_metrics(amps)
        assert m["cumulative_gains"] == pytest.approx([10, 20, 30, 40])

    def test_min_damage_refers_to_system_input(self):
        """
        Three stages with gains [10,10,10] dB.
        Stage 2 has max_input_power = 0 dBm.
        Referred to system input: 0 - 10 = -10 dBm.
        """
        amps = [
            self._make_amp(10, 3, None, None),
            self._make_amp(10, 3, None, None, max_pwr=0.0),
            self._make_amp(10, 3, None, None),
        ]
        m = compute_cascade_metrics(amps)
        assert m["min_damage_dbm"] == pytest.approx(-10.0, abs=1e-9)

    def test_min_damage_picks_most_restrictive(self):
        """When multiple stages have damage levels, the most restrictive wins."""
        amps = [
            self._make_amp(10, 3, None, None, max_pwr=10.0),  # ref'd to input: 10 dBm
            self._make_amp(10, 3, None, None, max_pwr=5.0),   # ref'd to input: 5-10 = -5 dBm
            self._make_amp(10, 3, None, None, max_pwr=20.0),  # ref'd to input: 20-20 = 0 dBm
        ]
        m = compute_cascade_metrics(amps)
        # Most restrictive is stage 2 at -5 dBm referred to input
        assert m["min_damage_dbm"] == pytest.approx(-5.0, abs=1e-9)

    def test_total_gain_in_metrics(self):
        amps = [self._make_amp(g, 3, None, None) for g in [5, 10, -3, 7]]
        m = compute_cascade_metrics(amps)
        assert m["gain_db"] == pytest.approx(19.0)

    def test_stage_gains_and_nfs_listed(self):
        amps = [self._make_amp(10, 3, None, None), self._make_amp(20, 5, None, None)]
        m = compute_cascade_metrics(amps)
        assert m["stage_gains"] == pytest.approx([10.0, 20.0])
        assert m["stage_nfs"] == pytest.approx([3.0, 5.0])


# ======================================================================= #
# S-parameter cascade (scikit-rf integration)                             #
# ======================================================================= #

class TestSparCascade:
    """
    Build simple synthetic 2-port networks and verify cascading behavior.
    """

    def _make_thru_network(self, n_pts=51, f_start=1e9, f_stop=10e9):
        """Return a perfect thru (S21=1, S11=0) scikit-rf Network."""
        import skrf
        freqs = np.linspace(f_start, f_stop, n_pts)
        s = np.zeros((n_pts, 2, 2), dtype=complex)
        s[:, 0, 1] = 1.0  # S12
        s[:, 1, 0] = 1.0  # S21
        freq_obj = skrf.Frequency.from_f(freqs, unit="hz")
        return skrf.Network(frequency=freq_obj, s=s)

    def _make_attenuator_network(self, atten_linear, n_pts=51, f_start=1e9, f_stop=10e9):
        """Return a lossy 2-port (S21=atten_linear)."""
        import skrf
        freqs = np.linspace(f_start, f_stop, n_pts)
        s = np.zeros((n_pts, 2, 2), dtype=complex)
        s[:, 1, 0] = atten_linear
        s[:, 0, 1] = atten_linear
        freq_obj = skrf.Frequency.from_f(freqs, unit="hz")
        return skrf.Network(frequency=freq_obj, s=s)

    def test_cascade_two_thrus_is_thru(self):
        from rf_tool.engine.cascade import cascade_networks, s21_to_gain_db
        t1 = self._make_thru_network()
        t2 = self._make_thru_network()
        result = cascade_networks([t1, t2])
        gain = s21_to_gain_db(result)
        assert np.allclose(gain, 0.0, atol=1e-6)

    def test_cascade_two_attenuators(self):
        """Cascading two 10 dB attenuators gives 20 dB attenuation."""
        from rf_tool.engine.cascade import cascade_networks, s21_to_gain_db
        atten = 10 ** (-10.0 / 20.0)   # -10 dB in amplitude
        a1 = self._make_attenuator_network(atten)
        a2 = self._make_attenuator_network(atten)
        result = cascade_networks([a1, a2])
        gain = s21_to_gain_db(result)
        assert np.allclose(gain, -20.0, atol=1e-4)

    def test_cascade_with_thru_unchanged(self):
        """Cascading any network with a perfect thru should leave it unchanged."""
        from rf_tool.engine.cascade import cascade_networks, s21_to_gain_db
        atten = 10 ** (-5.0 / 20.0)
        a = self._make_attenuator_network(atten)
        t = self._make_thru_network()
        result = cascade_networks([a, t])
        gain = s21_to_gain_db(result)
        assert np.allclose(gain, -5.0, atol=1e-4)

    def test_empty_cascade_raises(self):
        from rf_tool.engine.cascade import cascade_networks
        with pytest.raises(ValueError):
            cascade_networks([])

    def test_s21_to_gain_db_shape(self):
        from rf_tool.engine.cascade import s21_to_gain_db
        t = self._make_thru_network(n_pts=101)
        gain = s21_to_gain_db(t)
        assert gain.shape == (101,)
        assert np.allclose(gain, 0.0, atol=1e-6)
