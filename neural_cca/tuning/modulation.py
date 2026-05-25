"""Modulation ratio and cross-orientation suppression analyses.

Provides per-orientation F1/F0 modulation ratios and a cross-orientation
suppression index.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._utils import circ_dist, circ_mean, guarded_divide, wrap180
from ._filter import _build_trial_filter, _TrialFilteredSpikes
from .tuning import compute_f0_f1_f2

__all__ = [
    "modulation_ratio_per_orientation",
    "cross_orientation_suppression",
]


def modulation_ratio_per_orientation(
    spike_times: npt.NDArray[np.float64],
    trials: npt.NDArray[np.int64],
    angles: npt.NDArray[np.float64],
    bin_size: float = 0.05,
    stim_window: tuple[float, float] = (0.5, 2.5),
    stim_frequency: float = 2.0,
    cluster_labels: npt.NDArray[np.int64] | None = None,
    cluster_id: int | None = None,
    *,
    _filter: _TrialFilteredSpikes | None = None,
) -> dict[float, float]:
    """F1/F0 modulation ratio computed separately for each orientation.

    Simple cells typically have F1/F0 > 1 at their preferred orientation,
    while complex cells have F1/F0 < 1.

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        angles: Stimulus angle per trial (degrees).
        bin_size: PSTH bin width (seconds).
        stim_window: ``(onset, end)`` of the stimulus period within
            each trial (seconds). PSTHs are built on this window.
        stim_frequency: Temporal frequency of stimulus (Hz).
        cluster_labels: Cluster label per spike (optional).
        cluster_id: Cluster ID for per-cluster analysis.
        _filter: **Private** — pre-built per-trial filter from
            :func:`_build_trial_filter`.  When supplied, the per-trial
            spike-window filter is reused; only the per-orientation
            PSTH binning still happens here.  External callers should
            leave this ``None``.

    Returns:
        Dict mapping ``{angle_degrees: f1_f0_ratio}``.
    """
    if _filter is None:
        _filter = _build_trial_filter(
            spike_times,
            trials,
            angles,
            stim_window=stim_window,
            cluster_labels=cluster_labels,
            cluster_id=cluster_id,
        )

    angles = _filter.angles
    psth_fs = 1.0 / bin_size
    # Use ``ceil + linspace`` instead of ``np.arange`` with a float
    # step so the bin count is independent of floating-point rounding
    # of ``stim_offset - stim_onset``.
    n_bins = int(np.ceil((_filter.stim_offset - _filter.stim_onset) / bin_size))
    bins = np.linspace(_filter.stim_onset, _filter.stim_offset, n_bins + 1)
    unique_angles = np.unique(angles)
    result: dict[float, float] = {}

    for angle in unique_angles:
        angle_trials = np.where(angles == angle)[0]
        psths = []
        for t in angle_trials:
            spk = _filter.spike_times_by_trial[int(t)]
            if len(spk) == 0:
                psths.append(np.zeros(len(bins) - 1))
            else:
                counts, _ = np.histogram(spk, bins=bins)
                psths.append(counts / bin_size)

        if not psths:
            result[float(angle)] = np.nan
            continue

        psth_array = np.array(psths)
        F0, F1, _F2 = compute_f0_f1_f2(psth_array, psth_fs, stim_frequency)
        mean_f0 = float(F0.mean())
        mean_f1 = float(F1.mean())
        result[float(angle)] = guarded_divide(mean_f1, mean_f0)

    return result


def cross_orientation_suppression(
    responses: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
) -> float:
    r"""Cross-orientation suppression index.

    Estimates how much the orthogonal orientation suppresses the
    preferred-orientation response:

    .. math::

        \mathrm{COS} = 1 - \frac{R_\text{orth}}{R_\text{pref}}

    This is a proxy measure computed from the tuning curve.  A true
    COS measurement requires a plaid stimulus protocol (preferred +
    orthogonal superimposed).

    Values near 1 indicate strong suppression; values near 0 indicate
    the orthogonal response is comparable to the preferred response.
    Negative values indicate facilitation.

    Args:
        responses: Mean firing rates at each direction.
        angles: Stimulus angles in degrees.

    Returns:
        COS proxy index (float).
    """
    responses = np.asarray(responses, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)

    # Preferred orientation via vector sum (period 180°), not the
    # noisiest single bin via ``argmax``.  Mirrors the convention used
    # by ``preferred_dori``, ``gosi``, and ``gdsi``.
    pref_angle = circ_mean(angles, weights=responses, period=180.0)
    if np.isnan(pref_angle):
        return np.nan
    pref_idx = int(np.argmin(circ_dist(angles, pref_angle, period=180.0)))
    r_pref = float(responses[pref_idx])

    if r_pref == 0:
        # COS = 1 - R_orth/R_pref is undefined when R_pref = 0.
        # NaN reflects "cannot compute" rather than "no suppression".
        return np.nan

    # Find orthogonal (±90°) using circular distance with the
    # orientation period (180°). Linear distance picks the wrong sample
    # whenever the orthogonal target sits across the 0°/180° seam, and
    # the default 360° period would treat orientation wraparound as if
    # 0° and 180° were distinct.
    orth_angles = [wrap180(pref_angle + 90), wrap180(pref_angle - 90)]
    orth_responses = []
    for oa in orth_angles:
        idx = int(np.argmin(circ_dist(angles, oa, period=180.0)))
        orth_responses.append(responses[idx])
    r_orth = float(np.mean(orth_responses))

    return 1.0 - r_orth / r_pref
