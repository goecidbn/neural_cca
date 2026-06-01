"""Statistical testing for orientation selectivity.

Provides permutation tests, ANOVA, and bootstrap confidence intervals
for tuning analysis metrics.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import f_oneway, norm

from .._utils import make_rng
from ._filter import _build_trial_filter, _TrialFilteredSpikes
from .selectivity import _rayleigh_test, dosi_circular_normalised

__all__ = [
    "orientation_selectivity_significance",
    "anova_across_orientations",
    "bootstrap_ci",
    "bootstrap_ci_strata",
]


# ---------------------------------------------------------------------------
# Bootstrap CI helpers
# ---------------------------------------------------------------------------


def _percentile_ci(
    boot_stats: npt.NDArray,
    ci_level: float,
) -> tuple[float, float]:
    """Plain percentile CI from a bootstrap distribution."""
    alpha = 1.0 - ci_level
    lo = float(np.nanpercentile(boot_stats, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


def _bca_ci(
    estimate: float,
    boot_stats: npt.NDArray,
    jackknife_stats: npt.NDArray,
    ci_level: float,
) -> tuple[float, float]:
    r"""Bias-corrected and accelerated (BCa) confidence interval.

    Computes the BCa percentile end-points :math:`\alpha_1`,
    :math:`\alpha_2` (Efron 1987) from the bootstrap distribution and
    the jackknife distribution, then reports the corresponding
    bootstrap quantiles.

    The bias correction :math:`z_0` shifts the percentile end-points
    by the fraction of bootstrap replicates below the point estimate;
    the acceleration :math:`\hat{a}` corrects for non-constant
    variance of the statistic across the parameter range.  Together
    they yield a CI that is *second-order accurate* and
    transformation-respecting, in contrast to the (first-order, not
    transformation-respecting) percentile CI.

    When the bootstrap distribution is degenerate (constant, or all
    above / below the point estimate) the BCa formulas blow up — the
    helper falls back to the plain percentile CI and emits no warning
    because the percentile CI is also undefined in that case.

    References:
        Efron, B. (1987).  *Better bootstrap confidence intervals*.
        Journal of the American Statistical Association 82(397),
        171–185.  doi:10.1080/01621459.1987.10478410.

        DiCiccio, T. J. & Efron, B. (1996).  *Bootstrap confidence
        intervals*.  Statistical Science 11(3), 189–212.
        doi:10.1214/ss/1032280214.
    """
    # NOTE: every degenerate path here returns ``(NaN, NaN)``.  The
    # outer ``bootstrap_ci`` / ``bootstrap_ci_strata`` detect the NaN
    # endpoints and fall back to the percentile CI while also
    # relabelling ``method="percentile"`` so the user-visible
    # ``"method"`` key truthfully reflects which CI was used.  An
    # earlier version of this helper returned ``_percentile_ci(...)``
    # inline on degeneracy, which produced finite endpoints but left
    # the outer label as ``"bca"`` — the answer was correct but the
    # provenance was a lie.  Keep the NaN-sentinel convention.
    alpha = 1.0 - ci_level
    boot_valid = boot_stats[~np.isnan(boot_stats)]
    if len(boot_valid) < 2:
        return float("nan"), float("nan")

    # z0: bias correction — Φ⁻¹ of the fraction of bootstrap replicates
    # below the point estimate.  Degenerate ratios (0 or 1) make
    # ``norm.ppf`` return ±inf; flag via NaN so the outer caller can
    # fall back to plain percentile *and* relabel ``"method"``.
    below = float(np.mean(boot_valid < estimate))
    if not 0.0 < below < 1.0:
        return float("nan"), float("nan")
    z0 = float(norm.ppf(below))

    # a: acceleration — skewness of the jackknife distribution.
    jk_valid = jackknife_stats[~np.isnan(jackknife_stats)]
    if len(jk_valid) < 2:
        return float("nan"), float("nan")
    jk_mean = float(np.mean(jk_valid))
    # Acceleration uses ``diffs = jk_mean - jk_valid`` per DiCiccio &
    # Efron (1996, Stat. Sci. 11:189) eq. 2.13.  The opposite sign
    # ``(jk_valid - jk_mean)`` is common in textbook write-ups; both
    # give the *same* ``a`` because the numerator is sum-of-cubes and
    # the denominator is sum-of-squares^(3/2) — the cube preserves the
    # sign on top, the 3/2-power of squared values is positive, and
    # the overall sign survives unchanged.  Leaving this comment so
    # future reviewers don't second-guess the sign against a textbook.
    diffs = jk_mean - jk_valid
    num = float(np.sum(diffs**3))
    den = 6.0 * (float(np.sum(diffs**2))) ** 1.5
    # Tighten the degeneracy guard: an exact ``den == 0.0`` check lets
    # subnormal denominators slip through and amplify ``a`` to extreme
    # values that drive the downstream ``alpha_lo``/``alpha_hi``
    # computation off-rails.  Comparing against the float64 underflow
    # floor (~1e-300) catches the same "all jackknife diffs zero" case
    # without permitting a near-zero-but-not-exactly-zero blow-up.
    if abs(den) < 1e-300:
        return float("nan"), float("nan")
    a = num / den

    z_lo = float(norm.ppf(alpha / 2))
    z_hi = float(norm.ppf(1 - alpha / 2))
    alpha_lo = float(norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo))))
    alpha_hi = float(norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi))))

    # Guard against degenerate BCa end-points (e.g. alpha_lo > alpha_hi
    # when a is very negative); NaN-sentinel triggers outer percentile
    # fallback with correct ``"method"`` labelling.
    if not 0.0 <= alpha_lo < alpha_hi <= 1.0:
        return float("nan"), float("nan")

    lo = float(np.nanpercentile(boot_stats, 100 * alpha_lo))
    hi = float(np.nanpercentile(boot_stats, 100 * alpha_hi))
    return lo, hi


def orientation_selectivity_significance(
    responses: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    n_permutations: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> dict:
    r"""Test significance of orientation selectivity via permutation.

    The canonical V1-literature significance test (Mazurek et al. 2014;
    Niell & Stryker 2008): shuffle the (rate, angle) pairing
    *n_permutations* times, recompute OSI on each shuffle, and report
    the tail probability of the observed OSI under the resulting null
    distribution.  This is the **non-parametric standard** for tuning
    significance — unlike a parametric Rayleigh it makes no IID
    assumption on the rate observations and adapts automatically to
    the cell's actual noise structure.

    The *p*-value uses the Phipson & Smyth (2010) ``+1`` correction:

    .. math::

        \hat{p} = \frac{1 + \#\{T_b \geq T_\text{obs}\}}{1 + B}

    so the smallest reportable value is :math:`1/(B+1)` instead of
    ``0``.  The previous estimator could return :math:`\hat{p} = 0`
    even when the true tail probability is non-zero, which is
    formally wrong and confuses downstream FDR procedures (an exact
    ``0`` cannot be FDR-adjusted).

    A Rayleigh statistic on the **doubled-angle, rate-weighted**
    representation is still reported as ``p_rayleigh`` for legacy
    consumers, but it is **non-standard** for tuning-curve
    significance: a Rayleigh test assumes IID angle observations from
    a single circular distribution, whereas a tuning curve is "rates
    conditional on angle" with within-condition trial variance.  See
    the function docstring of
    :func:`~neural_cca.tuning.selectivity._rayleigh_test` for details.
    Treat ``p_rayleigh`` as a descriptive concentration statistic,
    not a calibrated tail probability.  ``is_significant`` now keys
    off the permutation test alone.

    Args:
        responses: Mean firing rates at each orientation.
        angles: Stimulus orientations in degrees.
        n_permutations: Number of permutation resamples.
        rng: ``numpy.random.Generator``, integer seed, or ``None``.

    Returns:
        Dict with keys:

        - ``"osi"`` — observed OSI
        - ``"p_permutation"`` — permutation *p*-value with the
          Phipson & Smyth ``+1`` correction
        - ``"p_rayleigh"`` — rate-weighted Rayleigh statistic
          (descriptive; *not* a calibrated tail probability — see
          above)
        - ``"is_significant"`` — ``True`` if ``p_permutation < 0.05``

    References:
        Mazurek, M., Kager, M. & Van Hooser, S. D. (2014).  *Robust
        quantification of orientation selectivity and direction
        selectivity*.  Frontiers in Neural Circuits 8:92.
        doi:10.3389/fncir.2014.00092.

        Phipson, B. & Smyth, G. K. (2010).  *Permutation p-values
        should never be zero: calculating exact p-values when
        permutations are randomly drawn*.  Statistical Applications
        in Genetics and Molecular Biology 9(1), Article 39.
        doi:10.2202/1544-6115.1585.

        Niell, C. M. & Stryker, M. P. (2008).  *Highly selective
        receptive fields in mouse visual cortex*.  Journal of
        Neuroscience 28(30), 7520–7536.
        doi:10.1523/JNEUROSCI.0623-08.2008.
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

    # Phipson & Smyth (2010) ``+1`` correction: avoids the
    # ``p_perm = 0`` artefact for strongly-tuned cells.  The smallest
    # value is now 1 / (n_permutations + 1).
    #
    # The ``+1`` in both numerator and denominator simulates *including
    # the observed value in the null distribution*, which is the
    # textbook discrete-uniform formulation.  Note that the observed
    # statistic itself is NOT inserted into ``null_osis``; it is only
    # accounted for via the correction term.  Treating the null as
    # ``[null_osis, observed_osi]`` and dividing by ``1 + n_permutations``
    # gives the same value without the extra array allocation.
    n_ge = int(np.sum(null_osis >= observed_osi))
    p_perm = (1 + n_ge) / (1 + n_permutations)

    # Rate-weighted Rayleigh — kept for backwards compatibility but
    # explicitly *not* claimed to be a calibrated p-value any more.
    # See the docstring of ``orientation_selectivity_significance`` and
    # of ``_rayleigh_test`` for the rationale.
    angles_rad = np.deg2rad(angles)
    p_rayleigh = _rayleigh_test(2.0 * angles_rad, responses)

    return {
        "osi": float(observed_osi),
        "p_permutation": p_perm,
        "p_rayleigh": p_rayleigh,
        "is_significant": p_perm < 0.05,
    }


def anova_across_orientations(
    spike_times: npt.NDArray[np.float64],
    trials: npt.NDArray[np.int64],
    angles: npt.NDArray[np.float64],
    stim_window: tuple[float, float] | None = None,
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
            each trial (seconds; **required**, no portable default).
            Spikes inside the half-open interval ``[onset, end)``
            contribute to the per-trial firing rate.
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
            spike_times,
            trials,
            angles,
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
    *,
    method: str = "bca",
) -> dict:
    r"""Bootstrap confidence interval for a *single-distribution* statistic.

    Resamples *data* with replacement, applies *stat_func* each time,
    and reports a confidence interval.  ``stat_func`` must be a
    function of one i.i.d. sample only — for example ``np.mean``,
    ``np.std``, or any statistic that does not depend on a paired
    label/condition.

    **Methods.**

    - ``method="bca"`` (default, recommended) — bias-corrected and
      accelerated CI (Efron 1987).  Second-order accurate and
      transformation-respecting; the right choice for skewed or
      boundary-bounded statistics like OSI / DSI in ``[0, 1]``.  Costs
      one extra ``stat_func`` evaluation per data point (the
      jackknife) which is negligible against ``n_bootstrap``
      evaluations.
    - ``method="percentile"`` — plain percentile CI.  First-order
      accurate, not transformation-respecting; for symmetric
      unbounded statistics (e.g. differences of means) the two
      methods agree to a few percent.  Retained for backwards
      compatibility and for cases where you explicitly want the
      simpler estimator.

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
        method: ``"bca"`` (default, second-order accurate) or
            ``"percentile"`` (first-order, legacy).

    Returns:
        Dict with keys:

        - ``"estimate"`` — point estimate (stat_func on original data)
        - ``"ci_lower"`` — lower bound of CI
        - ``"ci_upper"`` — upper bound of CI
        - ``"se"`` — standard error of bootstrap distribution
        - ``"method"`` — which CI method was used (``"bca"`` /
          ``"percentile"``; falls back to ``"percentile"`` if BCa
          could not be computed)

    References:
        Efron, B. (1987).  *Better bootstrap confidence intervals*.
        JASA 82(397), 171–185.  doi:10.1080/01621459.1987.10478410.

        DiCiccio, T. J. & Efron, B. (1996).  *Bootstrap confidence
        intervals*.  Statistical Science 11(3), 189–212.
        doi:10.1214/ss/1032280214.

        Carpenter, J. & Bithell, J. (2000).  *Bootstrap confidence
        intervals: when, which, what?  A practical guide for medical
        statisticians*.  Statistics in Medicine 19(9), 1141–1164.
    """
    if method not in ("bca", "percentile"):
        raise ValueError(f"method must be 'bca' or 'percentile', got {method!r}.")
    data = np.asarray(data, dtype=np.float64)
    rng = make_rng(rng)

    estimate = float(stat_func(data))

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = stat_func(sample)

    # Drop failed iterations (degenerate resamples can leave NaNs).
    valid = boot_stats[~np.isnan(boot_stats)]
    if len(valid) < 2:
        return {
            "estimate": estimate,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "se": np.nan,
            "method": method,
        }

    if method == "bca":
        # Jackknife distribution: leave-one-out estimates of the
        # statistic.  Used for the acceleration term.
        n = len(data)
        jackknife_stats = np.empty(n, dtype=np.float64)
        for i in range(n):
            jackknife_stats[i] = stat_func(np.delete(data, i))
        ci_lower, ci_upper = _bca_ci(estimate, boot_stats, jackknife_stats, ci_level)
        used_method = "bca"
        # Sentinel for fallback: if BCa returned NaN end-points
        # because of a degenerate distribution, fall back gracefully.
        if not (np.isfinite(ci_lower) and np.isfinite(ci_upper)):
            ci_lower, ci_upper = _percentile_ci(boot_stats, ci_level)
            used_method = "percentile"
    else:
        ci_lower, ci_upper = _percentile_ci(boot_stats, ci_level)
        used_method = "percentile"

    se = float(np.nanstd(boot_stats, ddof=1))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "method": used_method,
    }


def bootstrap_ci_strata(
    data: npt.ArrayLike,
    strata: npt.ArrayLike,
    stat_func: Callable[[npt.NDArray, npt.NDArray], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: np.random.Generator | int | None = None,
    *,
    method: str = "bca",
) -> dict:
    r"""Stratified bootstrap CI for a (data, label) statistic.

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

    See :func:`bootstrap_ci` for the ``method`` argument: ``"bca"``
    (default) gives a second-order-accurate, transformation-respecting
    interval (Efron 1987), which is what you want for boundary-bounded
    statistics like OSI / DSI / gOSI / gDSI in ``[0, 1]``.  The BCa
    jackknife here is performed on the *trials* (one trial held out
    at a time) so it remains consistent with the stratified resample.

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
        method: ``"bca"`` (default, recommended for OSI/DSI in
            ``[0, 1]``) or ``"percentile"``.

    Returns:
        Dict with keys ``"estimate"``, ``"ci_lower"``, ``"ci_upper"``,
        ``"se"``, ``"method"``. ``ci_lower`` / ``ci_upper`` / ``se``
        are NaN if fewer than two iterations produced finite values.

    Raises:
        ValueError: If *data* and *strata* have different lengths, or
            if *method* is unknown.

    References:
        Efron, B. (1987).  *Better bootstrap confidence intervals*.
        JASA 82(397), 171–185.  doi:10.1080/01621459.1987.10478410.

        Mazurek, M., Kager, M. & Van Hooser, S. D. (2014).  *Robust
        quantification of orientation selectivity and direction
        selectivity*.  Frontiers in Neural Circuits 8:92.
        doi:10.3389/fncir.2014.00092.
    """
    if method not in ("bca", "percentile"):
        raise ValueError(f"method must be 'bca' or 'percentile', got {method!r}.")
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

    # Small-stratum warning.  When the smallest stratum has fewer than
    # 3 observations, the BCa acceleration term (jackknife-derived) is
    # noisy and can return degenerate end-points that fall back to the
    # plain percentile CI.  Mazurek et al. (2014) recommend >= 5
    # repeats per condition for OS/DS bootstrapping; flag anything
    # below 3 explicitly so the caller knows the CI may be unstable.
    min_stratum_size = min(len(idx) for idx in stratum_indices)
    if min_stratum_size < 3:
        warnings.warn(
            f"Stratified bootstrap with smallest stratum size {min_stratum_size} < 3; "
            "BCa acceleration term may be unstable. Consider >=5 repeats per condition "
            "(Mazurek et al. 2014).",
            RuntimeWarning,
            stacklevel=2,
        )

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
            "method": method,
        }

    if method == "bca":
        # Stratified jackknife: leave one trial out at a time.  Note
        # that the stratum sizes shrink by one for the affected
        # stratum; this is the standard handling.
        n = len(data)
        jackknife_stats = np.empty(n, dtype=np.float64)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            jackknife_stats[i] = stat_func(data[mask], strata[mask])
        ci_lower, ci_upper = _bca_ci(estimate, boot_stats, jackknife_stats, ci_level)
        used_method = "bca"
        if not (np.isfinite(ci_lower) and np.isfinite(ci_upper)):
            ci_lower, ci_upper = _percentile_ci(boot_stats, ci_level)
            used_method = "percentile"
    else:
        ci_lower, ci_upper = _percentile_ci(boot_stats, ci_level)
        used_method = "percentile"

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.nanstd(boot_stats, ddof=1)),
        "method": used_method,
    }
