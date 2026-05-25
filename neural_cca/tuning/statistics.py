"""Statistical testing for orientation selectivity.

Provides permutation tests, ANOVA, and bootstrap confidence intervals
for tuning analysis metrics.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import f_oneway

from ._filter import _TrialFilteredSpikes, _build_trial_filter
from .selectivity import dosi_circular_normalised, _rayleigh_test
from .._utils import make_rng

__all__ = [
    "orientation_selectivity_significance",
    "anova_across_orientations",
    "bootstrap_ci",
    "bootstrap_ci_strata",
]


def orientation_selectivity_significance(
    responses: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    n_permutations: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """Test significance of orientation selectivity.

    Combines a permutation test (shuffle angle–response pairings and
    recompute OSI) with a Rayleigh test for circular non-uniformity.

    Args:
        responses: Mean firing rates at each orientation.
        angles: Stimulus orientations in degrees.
        n_permutations: Number of permutation resamples.
        rng: ``numpy.random.Generator``, integer seed, or ``None``.

    Returns:
        Dict with keys:

        - ``"osi"`` — observed OSI
        - ``"p_permutation"`` — permutation-test *p*-value
        - ``"p_rayleigh"`` — Rayleigh-test *p*-value
        - ``"is_significant"`` — ``True`` if both *p* < 0.05
    """
    responses = np.asarray(responses, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    rng = make_rng(rng)

    observed_osi = dosi_circular_normalised(responses, angles)

    # Permutation test.
    #
    # IMPORTANT: shuffling *responses* while keeping *angles* fixed is
    # **intentional** here and is exactly what a permutation null
    # requires.  The null hypothesis is "rates are independent of
    # angles", and we sample from it by breaking the rate-angle pairing
    # on purpose.  Each iteration draws a fresh assignment of rates to
    # the original (fixed) angles and recomputes the statistic; the
    # tail probability of the observed value under that distribution
    # is the p-value.
    #
    # Do NOT "fix" this to a stratified resample (as is done for the
    # bootstrap CI in :func:`bootstrap_ci_strata`).  A stratified
    # resample preserves the (rate, angle) pairing, which is the right
    # thing for confidence-interval estimation but the wrong thing for
    # the null distribution: it would never break the association we
    # are testing against.
    null_osis = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(responses)
        null_osis[i] = dosi_circular_normalised(shuffled, angles)

    p_perm = float(np.mean(null_osis >= observed_osi))

    # Rayleigh test
    angles_rad = np.deg2rad(angles)
    p_rayleigh = _rayleigh_test(2.0 * angles_rad, responses)

    return {
        "osi": float(observed_osi),
        "p_permutation": p_perm,
        "p_rayleigh": p_rayleigh,
        "is_significant": p_perm < 0.05 and p_rayleigh < 0.05,
    }


def anova_across_orientations(
    spike_times: npt.NDArray[np.float64],
    trials: npt.NDArray[np.int64],
    angles: npt.NDArray[np.float64],
    stim_window: tuple[float, float] = (0.5, 2.5),
    cluster_labels: npt.NDArray[np.int64] | None = None,
    cluster_id: int | None = None,
    *,
    _filter: _TrialFilteredSpikes | None = None,
) -> dict:
    """One-way ANOVA of firing rates across orientations.

    Tests whether mean firing rate differs significantly between
    stimulus orientations.

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        angles: Stimulus angle per trial (degrees).
        stim_window: ``(onset, end)`` of the stimulus period within
            each trial (seconds). Spikes inside this window contribute
            to the per-trial firing rate.
        cluster_labels: Cluster label per spike (optional).
        cluster_id: Cluster ID for per-cluster analysis.
        _filter: **Private** — pre-built per-trial filter from
            :func:`_build_trial_filter`.  When supplied, the function
            skips the per-trial rebuild and reuses the existing object;
            this is how :func:`get_os_metrics` avoids walking the
            spike arrays a second time.  External callers should pass
            the raw arrays and leave this ``None``.

    Returns:
        Dict with keys:

        - ``"f_stat"`` — F-statistic
        - ``"p_value"`` — ANOVA *p*-value
        - ``"group_means"`` — dict of ``{angle: mean_rate}``
        - ``"group_stds"`` — dict of ``{angle: std_rate}``
    """
    if _filter is None:
        _filter = _build_trial_filter(
            spike_times, trials, angles,
            stim_window=stim_window,
            cluster_labels=cluster_labels,
            cluster_id=cluster_id,
        )

    angles = _filter.angles
    mfrs = _filter.mfrs
    unique_angles = np.unique(angles)
    groups: dict[float, list[float]] = {
        float(ang): mfrs[angles == ang].tolist() for ang in unique_angles
    }

    group_arrays = [np.array(v) for v in groups.values()]

    # Need at least 2 groups with data
    valid = [g for g in group_arrays if len(g) >= 2]
    if len(valid) < 2:
        return {
            "f_stat": np.nan,
            "p_value": np.nan,
            "group_means": {k: float(np.mean(v)) for k, v in groups.items()},
            "group_stds": {k: float(np.std(v)) for k, v in groups.items()},
        }

    f_stat, p_value = f_oneway(*group_arrays)

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "group_means": {k: float(np.mean(v)) for k, v in groups.items()},
        "group_stds": {k: float(np.std(v)) for k, v in groups.items()},
    }


def bootstrap_ci(
    data: npt.ArrayLike,
    stat_func: Callable[[npt.NDArray], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """Bootstrap confidence interval for a *single-distribution* statistic.

    Resamples *data* with replacement, applies *stat_func* each time,
    and reports the percentile confidence interval. ``stat_func`` must
    be a function of one i.i.d. sample only — for example
    ``np.mean``, ``np.std``, or any statistic that does not depend on
    a paired label/condition.

    .. note::

        Use this function when ``data`` is a flat sample drawn from a
        single distribution (e.g. ISI durations, peak amplitudes,
        across-trial firing rates of one cell at one stimulus). When
        each element of ``data`` is paired with a categorical label
        (e.g. trial firing rates labelled by stimulus angle) and the
        statistic depends on that pairing (e.g. OSI, DSI, gOSI),
        plain bootstrap shuffles the labels apart and produces a
        meaningless null. Use :func:`bootstrap_ci_strata` instead —
        it resamples within each label group so the (data, label)
        pairing is preserved.

    Args:
        data: 1-D data array.
        stat_func: Function mapping an array to a scalar statistic.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g. 0.95 for 95% CI).
        rng: ``numpy.random.Generator``, integer seed, or ``None``.

    Returns:
        Dict with keys:

        - ``"estimate"`` — point estimate (stat_func on original data)
        - ``"ci_lower"`` — lower bound of CI
        - ``"ci_upper"`` — upper bound of CI
        - ``"se"`` — standard error of bootstrap distribution
    """
    data = np.asarray(data, dtype=np.float64)
    rng = make_rng(rng)

    estimate = float(stat_func(data))

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = stat_func(sample)

    # Drop failed iterations (degenerate resamples can leave NaNs).
    # Without this guard, np.percentile and np.std propagate NaN and
    # the entire CI dict becomes NaN.
    alpha = 1.0 - ci_level
    valid = boot_stats[~np.isnan(boot_stats)]
    if len(valid) < 2:
        return {
            "estimate": estimate,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "se": np.nan,
        }
    ci_lower = float(np.nanpercentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))
    se = float(np.nanstd(boot_stats, ddof=1))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
    }


def bootstrap_ci_strata(
    data: npt.ArrayLike,
    strata: npt.ArrayLike,
    stat_func: Callable[[npt.NDArray, npt.NDArray], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """Stratified bootstrap CI for a (data, label) statistic.

    For each iteration, resamples *data* with replacement *within
    each stratum* (preserving the per-stratum sample size and the
    label of every position), then evaluates ``stat_func(resampled,
    strata)``. This is the appropriate bootstrap when the statistic
    is a function of paired observations and labels — e.g. orientation
    or direction selectivity, where each trial firing rate is paired
    with a stimulus angle.

    Resampling within strata preserves the joint structure of
    ``(data, strata)``: every value in ``resampled`` still corresponds
    to the same label that occupied its position in the original
    array. Plain :func:`bootstrap_ci` would draw across labels and
    destroy that pairing.

    Args:
        data: 1-D values, one per observation (e.g. trial firing
            rate). Must be the same length as *strata*.
        strata: Categorical label per observation (e.g. stimulus
            angle per trial).
        stat_func: Callable ``(data_resampled, strata) -> float``.
            The second argument is the *original* strata array; only
            ``data`` is resampled.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g. 0.95 for 95 % CI).
        rng: ``numpy.random.Generator``, integer seed, or ``None``.

    Returns:
        Dict with keys ``"estimate"``, ``"ci_lower"``, ``"ci_upper"``,
        ``"se"``. ``ci_lower`` / ``ci_upper`` / ``se`` are NaN if
        fewer than two iterations produced finite values.

    Raises:
        ValueError: If *data* and *strata* have different lengths.
    """
    data = np.asarray(data, dtype=np.float64)
    strata = np.asarray(strata)
    if data.shape != strata.shape:
        raise ValueError(
            f"data and strata must have the same shape; got "
            f"data.shape={data.shape}, strata.shape={strata.shape}"
        )

    rng = make_rng(rng)

    # Pre-compute the index arrays for each stratum once: avoids
    # rebuilding `np.where(strata == s)` on every bootstrap iteration.
    unique_strata = np.unique(strata)
    stratum_indices = [np.where(strata == s)[0] for s in unique_strata]

    estimate = float(stat_func(data, strata))

    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        resampled = np.empty_like(data)
        for idx in stratum_indices:
            # Resample within this stratum and write back to the
            # *same* positions, so `resampled` stays aligned with the
            # original `strata` array.
            resample_idx = rng.choice(idx, size=len(idx), replace=True)
            resampled[idx] = data[resample_idx]
        boot_stats[b] = stat_func(resampled, strata)

    valid = boot_stats[~np.isnan(boot_stats)]
    if len(valid) < 2:
        return {
            "estimate": estimate,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "se": np.nan,
        }

    alpha = 1.0 - ci_level
    return {
        "estimate": estimate,
        "ci_lower": float(np.nanpercentile(boot_stats, 100 * alpha / 2)),
        "ci_upper": float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2))),
        "se": float(np.nanstd(boot_stats, ddof=1)),
    }
