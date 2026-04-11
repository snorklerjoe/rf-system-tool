"""
RF Cascade Analysis Engine.

This module implements the mathematical core for cascading RF blocks:

Small-signal (linear) analysis
================================
* Friis noise figure formula for cascaded stages.
* S-parameter cascading via scikit-rf.

Large-signal (non-linear) scalar analysis
==========================================
* IP3 cascading using worst-case phase-aligned voltage addition.
* P1dB cascading using the same approach.

The two analyses are intentionally kept SEPARATE so that complex
S-parameter math is never mixed with scalar nonlinear math.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# We avoid importing RFBlock at module level to prevent circular imports.
# Type annotations use strings.


# ======================================================================= #
# Helper: dB ↔ linear conversions                                         #
# ======================================================================= #

def db_to_linear_power(db: float) -> float:
    """Convert dB value to linear power ratio."""
    return 10.0 ** (db / 10.0)


def linear_power_to_db(linear: float) -> float:
    """Convert linear power ratio to dB."""
    if linear <= 0:
        return -math.inf
    return 10.0 * math.log10(linear)


def dbm_to_mw(dbm: float) -> float:
    """Convert dBm to milliwatts."""
    return 10.0 ** (dbm / 10.0)


def mw_to_dbm(mw: float) -> float:
    """Convert milliwatts to dBm."""
    if mw <= 0:
        return -math.inf
    return 10.0 * math.log10(mw)


# ======================================================================= #
# Friis Noise Figure Cascade                                               #
# ======================================================================= #

def cascade_noise_figure(
    gains_db: Sequence[float],
    nfs_db: Sequence[float],
) -> float:
    """
    Compute cascaded noise figure using the Friis formula.

        F_total = F1 + (F2-1)/G1 + (F3-1)/(G1*G2) + ...

    Parameters
    ----------
    gains_db : sequence of float
        Linear power gains of each stage in dB.
    nfs_db : sequence of float
        Noise figures of each stage in dB.

    Returns
    -------
    float
        Total cascaded noise figure in dB.

    Notes
    -----
    Uses Friis's formula with exact linear arithmetic.
    """
    if len(gains_db) != len(nfs_db):
        raise ValueError("gains_db and nfs_db must have the same length")
    if not gains_db:
        raise ValueError("At least one stage required")

    # Convert to linear
    Gs = [db_to_linear_power(g) for g in gains_db]
    Fs = [db_to_linear_power(nf) for nf in nfs_db]

    F_total = Fs[0]
    cumulative_gain = Gs[0]
    for i in range(1, len(Fs)):
        F_total += (Fs[i] - 1.0) / cumulative_gain
        cumulative_gain *= Gs[i]

    return linear_power_to_db(F_total)


# ======================================================================= #
# Cascaded Gain                                                            #
# ======================================================================= #

def cascade_gain(gains_db: Sequence[float]) -> float:
    """
    Return total cascaded gain = sum of individual gains in dB.

    Parameters
    ----------
    gains_db : sequence of float

    Returns
    -------
    float
        Total gain in dB.
    """
    return sum(gains_db)


# ======================================================================= #
# IP3 Cascade (Worst-Case Phase-Aligned Voltage Addition)                 #
# ======================================================================= #

def cascade_iip3(
    gains_db: Sequence[float],
    iip3s_dbm: Sequence[Optional[float]],
) -> Optional[float]:
    """
    Compute cascaded input-referred IP3 using the worst-case formula.

    The worst-case assumption is that all IM3 voltages add in phase.
    In terms of linear power (mW) referenced to the system input:

        1 / sqrt(IIP3_total_mW) =
            sum_k  sqrt(G1 * G2 * ... * G(k-1))  /  sqrt(IIP3_k_mW)

    where k-1 = 0 for the first stage (cumulative gain = 1).

    This is equivalent to the voltage-domain formula:
        1 / V_IIP3_total = sum_k  (a1_1 * a1_2 * ... * a1_(k-1)) / V_IIP3_k

    Parameters
    ----------
    gains_db : sequence of float
        Power gain of each stage in dB.
    iip3s_dbm : sequence of float or None
        Input-referred IP3 of each stage in dBm.
        None indicates an ideal (infinite IP3) stage.

    Returns
    -------
    float or None
        Cascaded IIP3 at system input in dBm, or None if all stages are ideal.

    Notes
    -----
    Stages with None IIP3 are skipped (treated as ideal).
    """
    if len(gains_db) != len(iip3s_dbm):
        raise ValueError("gains_db and iip3s_dbm must have the same length")

    Gs = [db_to_linear_power(g) for g in gains_db]

    sum_term = 0.0
    cumulative_gain = 1.0   # starts at 1 (no gain before stage 0)

    for i, iip3_dbm in enumerate(iip3s_dbm):
        if iip3_dbm is not None:
            iip3_mw = dbm_to_mw(iip3_dbm)
            # voltage-referenced sum: 1/V_IIP3_total = sum(cumulative_V_gain / V_IIP3_k)
            # In power: 1/sqrt(IIP3_total_mW) = sum(sqrt(cum_G) / sqrt(IIP3_k_mW))
            sum_term += math.sqrt(cumulative_gain) / math.sqrt(iip3_mw)
        cumulative_gain *= Gs[i]

    if sum_term == 0.0:
        return None  # all stages ideal

    iip3_total_mw = 1.0 / (sum_term ** 2)
    return mw_to_dbm(iip3_total_mw)


def cascade_oip3(
    gains_db: Sequence[float],
    iip3s_dbm: Sequence[Optional[float]],
) -> Optional[float]:
    """
    Compute cascaded output-referred IP3.

    OIP3_total = IIP3_total + total_gain_dB

    Parameters
    ----------
    gains_db : sequence of float
    iip3s_dbm : sequence of float or None

    Returns
    -------
    float or None
    """
    iip3 = cascade_iip3(gains_db, iip3s_dbm)
    if iip3 is None:
        return None
    total_gain = cascade_gain(gains_db)
    return iip3 + total_gain


# ======================================================================= #
# P1dB Cascade                                                             #
# ======================================================================= #

def cascade_p1db(
    gains_db: Sequence[float],
    p1db_out_dbm: Sequence[Optional[float]],
) -> Optional[float]:
    """
    Compute cascaded input-referred 1-dB compression point.

    Uses the same voltage-addition worst-case formula as IP3 but applied
    to P1dB directly.  Referred to the system input.

        1 / sqrt(P1dB_in_total_mW) =
            sum_k sqrt(G1*...*G(k-1)) / sqrt(P1dB_in_k_mW)

    where P1dB_in_k is the *input* P1dB of stage k = P1dB_out_k - gain_k.

    Parameters
    ----------
    gains_db : sequence of float
        Gain of each stage in dB.
    p1db_out_dbm : sequence of float or None
        Output-referred P1dB of each stage in dBm.  None = ideal.

    Returns
    -------
    float or None
        Input-referred cascaded P1dB in dBm.
    """
    if len(gains_db) != len(p1db_out_dbm):
        raise ValueError("gains_db and p1db_out_dbm must have the same length")

    Gs = [db_to_linear_power(g) for g in gains_db]
    sum_term = 0.0
    cumulative_gain = 1.0

    for i, p1db_dbm in enumerate(p1db_out_dbm):
        if p1db_dbm is not None:
            # Convert output P1dB to input P1dB for this stage
            p1db_in_dbm = p1db_dbm - gains_db[i]
            p1db_in_mw = dbm_to_mw(p1db_in_dbm)
            sum_term += math.sqrt(cumulative_gain) / math.sqrt(p1db_in_mw)
        cumulative_gain *= Gs[i]

    if sum_term == 0.0:
        return None

    p1db_total_mw = 1.0 / (sum_term ** 2)
    return mw_to_dbm(p1db_total_mw)


# ======================================================================= #
# Full Cascade Readout                                                     #
# ======================================================================= #

def compute_cascade_metrics(blocks: list) -> dict:
    """
    Compute full cascade metrics for a linear sequence of RFBlock objects.

    Parameters
    ----------
    blocks : list of RFBlock
        Ordered list of blocks from input to output.

    Returns
    -------
    dict with keys:
        "gain_db"       : total cascaded gain (dB)
        "nf_db"         : cascaded noise figure (dB)
        "iip3_dbm"      : input-referred cascaded IP3 (dBm or None)
        "oip3_dbm"      : output-referred cascaded IP3 (dBm or None)
        "p1db_in_dbm"   : input-referred cascaded P1dB (dBm or None)
        "min_damage_dbm": most restrictive max input power (dBm or None)
        "max_required_dbm": highest input floor from stage min-input limits, referred to system input (dBm or None)
        "stage_gains"   : per-stage gain (dB)
        "stage_nfs"     : per-stage noise figure (dB)
        "cumulative_gains": cumulative gain from input to end of each stage (dB)
    """
    if not blocks:
        return {
            "gain_db": 0.0,
            "nf_db": 0.0,
            "iip3_dbm": None,
            "oip3_dbm": None,
            "p1db_in_dbm": None,
            "min_damage_dbm": None,
            "max_required_dbm": None,
            "stage_gains": [],
            "stage_nfs": [],
            "cumulative_gains": [],
        }

    gains = [b.gain_db for b in blocks]
    nfs = [b.nf_db for b in blocks]

    # IIP3: convert OIP3 -> IIP3 per block
    iip3s = []
    for b in blocks:
        if b.iip3_dbm is not None:
            iip3s.append(b.iip3_dbm)
        else:
            iip3s.append(None)

    p1dbs_out = [b.p1db_dbm for b in blocks]

    # Minimum damage level (max input power) - referred to system input
    min_damage = None
    max_required = None
    cum_gain = 0.0
    for b in blocks:
        if b.max_input_power_dbm is not None:
            # Refer to system input: max_safe_input = b.max - cum_gain
            system_max = b.max_input_power_dbm - cum_gain
            if min_damage is None or system_max < min_damage:
                min_damage = system_max
        if b.min_input_power_dbm is not None:
            system_min = b.min_input_power_dbm - cum_gain
            if max_required is None or system_min > max_required:
                max_required = system_min
        cum_gain += b.gain_db

    # Cumulative gains
    cum_gains = []
    running = 0.0
    for g in gains:
        running += g
        cum_gains.append(running)

    return {
        "gain_db": cascade_gain(gains),
        "nf_db": cascade_noise_figure(gains, nfs),
        "iip3_dbm": cascade_iip3(gains, iip3s),
        "oip3_dbm": cascade_oip3(gains, iip3s),
        "p1db_in_dbm": cascade_p1db(gains, p1dbs_out),
        "min_damage_dbm": min_damage,
        "max_required_dbm": max_required,
        "stage_gains": gains,
        "stage_nfs": nfs,
        "cumulative_gains": cum_gains,
    }


# ======================================================================= #
# S-parameter Cascade (via scikit-rf)                                     #
# ======================================================================= #

def cascade_networks(networks: list):
    """
    Cascade a sequence of scikit-rf Network objects.

    Parameters
    ----------
    networks : list of skrf.Network
        Networks to cascade (must all be 2-port).

    Returns
    -------
    skrf.Network
        Cascaded network.
    """
    if not networks:
        raise ValueError("No networks provided")
    result = networks[0]
    for net in networks[1:]:
        result = result ** net
    return result


def s21_to_gain_db(network) -> np.ndarray:
    """
    Extract |S21| in dB from a scikit-rf Network.

    Parameters
    ----------
    network : skrf.Network

    Returns
    -------
    np.ndarray
        Gain in dB at each frequency point.
    """
    s21 = network.s[:, 1, 0]
    return 20.0 * np.log10(np.abs(s21) + 1e-300)
