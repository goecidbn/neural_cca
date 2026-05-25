"""Spike-sorting quality metrics.

Functions for evaluating spike-sorting quality: silhouette-based
misclassification, refractory-period violations, signal-to-noise
ratio, pre-stimulus spike counts, isolation distance, L-ratio,
d-prime, waveform stability, amplitude drift, and missing spikes.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_samples

__all__ = [
    "neg_silhouette_score",
    "spikes_before_stimulus",
    "est_snr",
    "calc_weighted_snr",
    "rpvs",
    "contamination_rate_hill",
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "d_prime_pairwise_matrix",
    "peak_amplitude_snr",
    "waveform_stability",
    "amplitude_drift",
    "fraction_missing",
]


def neg_silhouette_score(
    features: npt.ArrayLike,
    cluster_labels: npt.ArrayLike,
    relative: bool = True,
    all_clusters: bool = True,
    cluster_id: int | None = None,
    **kwargs: Any,
) -> float | int:
    """Proportion (or count) of samples with negative silhouette score.

    Args:
        features: Feature matrix, shape ``(n_samples, n_features)``.
            Use the same name across all feature-space metrics
            (:func:`isolation_distance`, :func:`l_ratio`,
            :func:`d_prime`).
        cluster_labels: Cluster label per sample.
        relative: Return ratio (``True``) or absolute count (``False``).
        all_clusters: Consider all clusters (``True``) or one *cluster_id*.
        cluster_id: Cluster ID to evaluate (required if
            ``all_clusters=False``).
        **kwargs: Forwarded to ``silhouette_samples``.

    Returns:
        Number or fraction of samples with silhouette < 0.

    Raises:
        ValueError: On invalid argument combinations.
    """
    _validate_cluster_args(all_clusters, cluster_id)

    scores = silhouette_samples(features, cluster_labels, **kwargs)

    if all_clusters:
        neg_count = int(np.sum(scores < 0))
        total = len(features)
    else:
        mask = np.asarray(cluster_labels) == cluster_id
        neg_count = int(np.sum(scores[mask] < 0))
        total = int(np.sum(mask))

    if relative:
        return neg_count / total if total > 0 else 0.0
    return neg_count


def spikes_before_stimulus(
    spike_times: np.ndarray,
    cluster_labels: np.ndarray,
    stim_onset: float,
    relative: bool = True,
    all_clusters: bool = True,
    cluster_id: int | None = None,
) -> float | int:
    """Count (or fraction) of spikes occurring before stimulus onset.

    Args:
        spike_times: Spike times (same units as *stim_onset*).
        cluster_labels: Cluster label per spike.
        stim_onset: Time of stimulus onset.
        relative: Return ratio (``True``) or absolute count (``False``).
        all_clusters: Consider all clusters (``True``) or one *cluster_id*.
        cluster_id: Cluster ID (required if ``all_clusters=False``).

    Returns:
        Number or fraction of pre-stimulus spikes.

    Raises:
        ValueError: On invalid argument combinations.
    """
    _validate_cluster_args(all_clusters, cluster_id)

    if all_clusters:
        n_before = int(np.sum(spike_times < stim_onset))
        total = len(spike_times)
    else:
        mask = cluster_labels == cluster_id
        n_before = int(np.sum(spike_times[mask] < stim_onset))
        total = int(np.sum(mask))

    if relative:
        return n_before / total if total > 0 else 0.0
    return n_before


def est_snr(waveforms: npt.NDArray) -> float:
    """Signal-to-noise ratio of a set of waveform snippets.

    ``SNR = (peak-to-peak of mean waveform) / (2 * std of residuals)``

    References:
        - https://doi.org/10.1109/TNSRE.2005.857687
        - https://doi.org/10.1152/jn.00680.2018

    Args:
        waveforms: Waveform matrix, shape ``(n_snippets, snippet_length)``.

    Returns:
        Estimated SNR (float).  ``np.nan`` when the noise standard
        deviation is *negligibly small* — specifically
        ``noise_std <= 1e-12 * max(sig_amp, 1.0)``.  The data-scale-
        relative threshold catches both genuine degeneracy (single
        snippet, identical traces) and the mean-subtraction rounding
        residual (~1e-15) that ``identical-rows`` input produces
        after `np.std`, which used to slip past an exact ``== 0``
        guard and yield a bogus ~1e15-scale SNR.
    """
    W_bar = np.mean(waveforms, axis=0, dtype=np.float64)
    sig_amp = float(np.max(W_bar) - np.min(W_bar))
    noise = waveforms - W_bar
    noise_std = float(np.std(noise))
    # Identical rows produce a residual ``noise_std`` of ~1e-15 (the
    # mean-subtraction rounding floor at float64), which slips past an
    # exact ``== 0`` guard and returns a bogus 1e15-scale SNR.  Compare
    # against the signal magnitude so the threshold tracks the data
    # scale instead of being a fixed absolute number.
    if noise_std <= 1e-12 * max(sig_amp, 1.0):
        return np.nan
    return float(sig_amp / (2.0 * noise_std))


def calc_weighted_snr(
    waveforms: npt.NDArray,
    cluster_labels: npt.NDArray,
) -> float:
    """Cluster-size-weighted SNR across all clusters.

    For each cluster, computes ``est_snr`` and weights it by the
    proportion of snippets belonging to that cluster.

    Clusters whose ``est_snr`` returns ``np.nan`` (degenerate noise:
    identical waveforms, or a single-snippet cluster) are **excluded**
    from the weighted mean and the remaining weights are renormalised.
    A ``RuntimeWarning`` is emitted whenever this happens so callers
    can spot the culprit cluster instead of seeing the entire recording
    silently reported as ``NaN``.  When **every** cluster is degenerate
    the function returns ``np.nan``.

    Args:
        waveforms: Waveform matrix, shape ``(n_snippets, snippet_length)``.
        cluster_labels: Cluster label per snippet.

    Returns:
        Weighted SNR (float).
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    weights = counts / np.sum(counts)

    snrs = np.array(
        [est_snr(waveforms[cluster_labels == cid]) for cid in unique],
        dtype=np.float64,
    )
    valid = ~np.isnan(snrs)
    # Always warn first when any cluster is degenerate — the user
    # should hear about it whether we can still return a partial
    # weighted mean (some clusters valid) or have to surrender (none
    # valid).  Burying this behind the early-return would silently
    # hide the very condition the function is meant to surface.
    if not valid.all():
        bad = [int(c) for c, ok in zip(unique, valid) if not ok]
        warnings.warn(
            f"calc_weighted_snr: cluster(s) {bad} had degenerate noise "
            "(identical waveforms or a single snippet); excluded from the "
            "weighted mean.  Remaining weights renormalised.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not valid.any():
        return float("nan")
    w = weights[valid] / weights[valid].sum()
    return float(np.dot(snrs[valid], w))


def rpvs(
    spike_times: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    refractory_period: float = 0.001,
    relative: bool = True,
    all_clusters: bool = True,
    cluster_id: int | None = None,
) -> float | int:
    """Refractory-period violations (RPVs).

    When ``all_clusters=True`` and *cluster_labels* are given, RPVs are
    summed across clusters (inter-cluster ISIs are not counted).
    Negative ISIs (trial-boundary artefacts) are excluded from the
    violation *count*.

    **Units.** *spike_times* and *refractory_period* must use the same
    time unit; the package convention everywhere is **seconds** (the
    default ``refractory_period=0.001`` is 1 ms).  If you pass
    millisecond spike times you must pass *refractory_period* in
    milliseconds too — there is no implicit conversion.

    **Definition of the relative rate.**  When ``relative=True`` this
    function returns

    .. math::

        \\text{rel\\_rpvs} = \\frac{N_\\text{violations}}{N_\\text{spikes}}

    i.e. the conventional Phy / Kilosort / SpikeInterface "violations
    per spike" metric.  This is *not* the same as
    ``violations / N_valid_ISIs``: trial-based recordings produce
    ``num_trials`` negative ISIs that are correctly excluded from the
    numerator, but the denominator is still the spike count so the
    value stays comparable to ratings from other spike sorters and
    stable across cluster size.  It is also *not* the formal Hill
    et al. (2011) contamination rate; multiply by ``2`` to get a
    rough estimate of the fraction of *spikes* involved in a violation.

    Args:
        spike_times: Spike times (seconds).
        cluster_labels: Cluster label per spike (optional for
            ``all_clusters=True`` without per-cluster splitting).
        refractory_period: Refractory period in seconds.
        relative: Return ratio of violations to total spikes (``True``)
            or absolute count (``False``).
        all_clusters: Evaluate all clusters (``True``) or one
            *cluster_id*.
        cluster_id: Cluster ID (required if ``all_clusters=False``).

    Returns:
        RPV count (int) or violations-per-spike fraction (float).

    Raises:
        ValueError: On invalid argument combinations, or when
            ``refractory_period`` is not strictly positive (a
            non-positive value silently inverts the
            ``isi < refractory`` comparison and reports zero
            violations for every spike train).
    """
    _validate_cluster_args(all_clusters, cluster_id)
    if not all_clusters and cluster_labels is None:
        raise ValueError("cluster_labels must be provided when all_clusters is False.")
    if refractory_period <= 0:
        raise ValueError(
            f"refractory_period must be positive (got {refractory_period}); "
            "a non-positive value silently inverts the violation comparison "
            "and reports zero RPVs regardless of the data."
        )

    # The denominator is the *spike count*, not the valid-ISI count.
    # Dividing by ``len(diffs)`` would inflate the value for trial-based
    # data with many trials and sparse clusters (each trial transition
    # contributes a negative ISI that drops out of the numerator AND
    # would drop out of the denominator, so the ratio would jump by a
    # factor of ``num_trials / num_spikes``).  ``count / N_spikes`` is
    # the conventional Phy / Kilosort / SpikeInterface RPV rate and
    # stays stable across cluster sizes and trial structure.
    if all_clusters and cluster_labels is None:
        diffs = np.diff(spike_times)
        diffs = diffs[diffs > 0]
        rpvs_count = int(np.sum(diffs < refractory_period))
        total = len(spike_times)

    elif all_clusters and cluster_labels is not None:
        rpvs_count = 0
        total = len(spike_times)
        for cid in np.unique(cluster_labels):
            spk = spike_times[cluster_labels == cid]
            diffs = np.diff(spk)
            diffs = diffs[diffs > 0]
            rpvs_count += int(np.sum(diffs < refractory_period))

    else:
        spk = spike_times[cluster_labels == cluster_id]
        diffs = np.diff(spk)
        diffs = diffs[diffs > 0]
        rpvs_count = int(np.sum(diffs < refractory_period))
        total = len(spk)

    if relative:
        if total == 0:
            return 0.0
        return rpvs_count / total
    return rpvs_count


# ---------------------------------------------------------------------------
# Hill 2011 contamination rate
# ---------------------------------------------------------------------------


def contamination_rate_hill(
    spike_times: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    recording_duration: float | None = None,
    refractory_period: float = 0.001,
    censored_period: float = 0.0,
    all_clusters: bool = True,
    cluster_id: int | None = None,
) -> float | dict[int, float]:
    r"""Estimated cluster contamination fraction (Hill et al. 2011).

    The Hill estimator converts an observed count of refractory-period
    violations into an estimate of the *false-positive fraction* of
    the cluster — i.e. the fraction of spikes that do **not** come from
    the target unit.  This is the calibrated metric reviewers expect
    when methods sections report e.g. "contamination < 10 %", and is
    the basis for the Allen Institute / IBL inclusion thresholds.

    .. math::

        C = \tfrac{1}{2}\left(
            1 - \sqrt{\,
                1 - \frac{2\,N_v\,T}{N^2\,(t_r - t_c)}
            }\,
        \right)

    where :math:`N_v` is the violation count, :math:`N` the number of
    spikes in the cluster, :math:`T` the recording duration in
    seconds, :math:`t_r` the refractory period in seconds, and
    :math:`t_c` the censored period (the dead time of the spike
    detector, typically the spike-sorting blanking window).

    The argument of the square root is clipped to :math:`[0, 1]`: if
    the observed violation count exceeds the level that a
    fully-random (50 % contamination) cluster would produce, the
    estimator saturates at :math:`C = 0.5` rather than returning
    ``nan`` from a negative discriminant.  This matches the
    SpikeInterface and `ecephys_spike_sorting` conventions.

    .. note::

        The estimator assumes contaminant spikes are temporally
        independent of the target spikes.  If the contaminant is
        another nearby unit whose firing is correlated with the
        target (cross-talk, common drive), the estimate is biased —
        anti-correlated contaminants are *under*-estimated,
        positively correlated contaminants are *over*-estimated.
        See [Llobet et al. 2022 / Sibille et al. 2024] for
        cross-contamination-aware refinements.

    .. warning::

        Two scaling pitfalls:

        - **Pass the total recording duration**, not the trial-window
          duration.  For a trial-based recording with ``n_trials``
          trials of length ``trial_dur`` seconds each, supply
          ``recording_duration = n_trials * trial_dur``.  Halving
          this value doubles the apparent contamination.
        - The default ``censored_period=0`` matches Hill's original
          formulation but slightly over-estimates contamination when
          the spike sorter blanks a window around each detection.
          Set ``censored_period`` to the sorter's blanking window
          (typically 0.5 ms for Kilosort-style sorters) for a
          tighter estimate.

    Args:
        spike_times: Spike times in seconds.
        cluster_labels: Optional cluster labels.  When given, the
            function returns one value per cluster (or one value for
            ``cluster_id`` when ``all_clusters=False``).
        recording_duration: Total recording duration in seconds.
            Required; an estimate from ``spike_times.max() -
            spike_times.min()`` is *not* used because it
            under-estimates for trial-based data (the window is at
            most one trial's length, but spikes live across many
            trials).
        refractory_period: Refractory period :math:`t_r` in seconds
            (default 1 ms).  Must be strictly positive.
        censored_period: Detector dead time :math:`t_c` in seconds
            (default 0 s).  Must satisfy ``0 <= censored_period <
            refractory_period``.
        all_clusters: Per-cluster dict (``True``) or single cluster
            (``False``).
        cluster_id: Cluster ID when ``all_clusters=False``.

    Returns:
        Estimated contamination fraction in ``[0, 0.5]``, or a
        ``{cluster_id: float}`` dict.  Returns ``np.nan`` for
        clusters with fewer than 2 spikes (rate cannot be estimated)
        or when ``recording_duration`` is non-positive.

    Raises:
        ValueError: When required arguments are missing or invalid:
            ``recording_duration`` not given, ``refractory_period``
            non-positive, ``censored_period`` outside
            ``[0, refractory_period)``.

    References:
        Hill, D. N., Mehta, S. B. & Kleinfeld, D. (2011).  *Quality
        metrics to accompany spike sorting of extracellular signals*.
        Journal of Neuroscience 31(24), 8699–8705.
        doi:10.1523/JNEUROSCI.0971-11.2011.

        Llobet, V., Wyngaard, A. & Barbour, B. (2022).  *Automatic
        post-processing and merging of multiple spike-sorting
        analyses with Lussac*.  bioRxiv 2022.02.08.479192.

        Sibille, J. et al. (2024).  *Assessing cross-contamination
        in spike-sorted electrophysiology data*.  eNeuro 11(8),
        ENEURO.0554-23.2024.  doi:10.1523/ENEURO.0554-23.2024.

        SpikeInterface `quality_metrics` module:
        https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics/isi_violations.html
    """
    if recording_duration is None:
        raise ValueError(
            "recording_duration is required.  For trial-based data "
            "pass n_trials * trial_duration, not the per-trial "
            "window — see the docstring warning."
        )
    if refractory_period <= 0:
        raise ValueError(f"refractory_period must be positive (got {refractory_period}).")
    if not (0.0 <= censored_period < refractory_period):
        raise ValueError(
            "censored_period must satisfy 0 <= censored_period < "
            f"refractory_period; got censored_period={censored_period}, "
            f"refractory_period={refractory_period}."
        )
    _validate_cluster_args(all_clusters, cluster_id)

    t_r = float(refractory_period)
    t_c = float(censored_period)
    T = float(recording_duration)
    # A non-positive duration makes the estimator undefined; surface NaN
    # uniformly via the per-cluster helper rather than building a
    # special-cased return-shape dispatcher up here.
    duration_valid = T > 0.0

    def _hill_one(spk: np.ndarray) -> float:
        if not duration_valid:
            return float("nan")
        N = int(spk.size)
        if N < 2:
            return float("nan")
        diffs = np.diff(np.sort(spk))
        diffs = diffs[diffs > 0]
        N_v = int(np.sum(diffs < t_r))
        # Closed-form Hill estimator.  ``disc`` (the radicand) can go
        # negative for catastrophically contaminated units; clip to 0
        # so the estimator saturates at C = 0.5 instead of returning
        # NaN from sqrt(negative).
        disc = 1.0 - (2.0 * N_v * T) / (N**2 * (t_r - t_c))
        disc = max(0.0, float(disc))
        return 0.5 * (1.0 - np.sqrt(disc))

    if cluster_labels is None:
        return _hill_one(np.asarray(spike_times, dtype=np.float64))

    cluster_labels = np.asarray(cluster_labels)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    if all_clusters:
        return {
            int(c): _hill_one(spike_times[cluster_labels == c]) for c in np.unique(cluster_labels)
        }
    return _hill_one(spike_times[cluster_labels == cluster_id])


# ---------------------------------------------------------------------------
# Helper: validate all_clusters / cluster_id combination
# ---------------------------------------------------------------------------


def _validate_cluster_args(
    all_clusters: bool,
    cluster_id: int | None,
) -> None:
    if all_clusters and cluster_id is not None:
        raise ValueError("Cannot specify 'cluster_id' when 'all_clusters' is True.")
    if not all_clusters and cluster_id is None:
        raise ValueError("Must specify 'cluster_id' when 'all_clusters' is False.")


def _mahalanobis_sq(
    features: npt.NDArray,
    mean: npt.NDArray,
    cov_inv: npt.NDArray,
) -> npt.NDArray:
    """Squared Mahalanobis distance of each row in *features* from *mean*."""
    diff = features - mean
    return np.sum(diff @ cov_inv * diff, axis=1)


def _ledoit_wolf_precision(features: npt.NDArray) -> npt.NDArray:
    r"""Ledoit–Wolf shrunk covariance, returned as its inverse (precision).

    The Ledoit–Wolf estimator (Ledoit & Wolf, 2004) returns a convex
    combination of the sample covariance :math:`S` and a scaled
    identity target :math:`F = \mu I` with :math:`\mu = \mathrm{tr}(S)/p`:

    .. math::

        \hat{\Sigma}_\mathrm{LW} = (1 - \rho^*)\,S + \rho^*\,F,
        \quad \rho^* \in [0, 1].

    The shrinkage intensity :math:`\rho^*` is *not* a free hyperparameter
    — it has a closed-form optimum

    .. math::

        \rho^* = \min\!\Bigl(1,\;
            \frac{\pi^* - \rho^*_S}{\gamma^* \cdot n}
        \Bigr),

    where :math:`\pi^*` estimates the sum of asymptotic variances of
    :math:`\sqrt{n}\,\mathrm{vec}(S - \Sigma)`, :math:`\rho^*_S`
    estimates its covariance with :math:`\mathrm{vec}(F - \Sigma)`,
    and :math:`\gamma^* = \lVert F - \Sigma \rVert_\mathrm{Fro}^2`
    measures how far the shrinkage target sits from the truth.
    The estimator therefore *adapts* :math:`\rho^*` to the data:
    abundant well-conditioned samples push :math:`\rho^* \to 0`
    (use :math:`S`); few samples or near-degenerate spectra push
    :math:`\rho^* \to 1` (use the identity target).  The result is
    the unique linear combination that minimises the expected
    Frobenius-norm error :math:`\mathbb{E}\lVert \hat{\Sigma} - \Sigma
    \rVert_\mathrm{Fro}^2` to first order in :math:`n^{-1}`.

    **Why this matters here.**  Spike-sorting feature matrices sit
    squarely in the small-:math:`n` / large-:math:`p` regime that
    breaks the sample covariance: a cluster of a few hundred spikes
    in a 12–32-dimensional PCA space gives :math:`n/p \approx 10`,
    so :math:`S` has the right rank but its smallest eigenvalues are
    biased toward zero by a factor :math:`(1 - p/n)^{-1}` and the
    precision matrix :math:`S^{-1}` blows them up.  Mahalanobis
    distances computed with :math:`S^{-1}` then over-weight the
    noisy directions and the resulting :func:`isolation_distance` /
    :func:`l_ratio` values jump around between clusters of similar
    quality.  Shrinking toward :math:`\mu I` regularises the
    spectrum without introducing a tunable parameter, which is the
    standard fix in the spike-sorting literature
    (e.g. Schmitzer-Torbert et al. 2005, Hill et al. 2011).

    The implementation delegates to
    :class:`sklearn.covariance.LedoitWolf` and returns the
    precision (``estimator.precision_``) directly, which is what the
    Mahalanobis-distance helpers consume.  Compared to the previous
    ``np.linalg.pinv(np.cov(X, rowvar=False))`` path, Ledoit–Wolf
    avoids silently discarding eigenvalues below the pinv tolerance
    (the discard threshold scales with the largest eigenvalue and
    therefore varies across clusters in a way that's invisible to
    the caller) and produces strictly positive-definite precision
    matrices for any non-trivial cluster.

    References:
        Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator
        for large-dimensional covariance matrices*. Journal of
        Multivariate Analysis, 88(2), 365–411.
        doi:10.1016/S0047-259X(03)00096-4.

        Schmitzer-Torbert, N., Jackson, J., Henze, D., Harris, K. &
        Redish, A. D. (2005). *Quantitative measures of cluster
        quality for use in extracellular recordings*. Neuroscience,
        131(1), 1–11.

        Hill, D. N., Mehta, S. B. & Kleinfeld, D. (2011).
        *Quality metrics to accompany spike sorting of
        extracellular signals*. Journal of Neuroscience, 31(24),
        8699–8705.
    """
    # ``store_precision=True`` makes sklearn cache the matrix-inverse
    # alongside the covariance, so a downstream Mahalanobis-distance
    # call reads ``estimator.precision_`` directly without a second
    # matrix inversion.  ``assume_centered=False`` lets the estimator
    # subtract the cluster mean internally — we hand it the raw
    # feature rows, not centred residuals.
    estimator = LedoitWolf(store_precision=True, assume_centered=False)
    estimator.fit(features)
    return estimator.precision_


# ---------------------------------------------------------------------------
# Isolation distance  (Harris et al. 2000)
# ---------------------------------------------------------------------------


def isolation_distance(
    features: npt.NDArray,
    cluster_labels: npt.NDArray,
    all_clusters: bool = True,
    cluster_id: int | None = None,
    *,
    mode: str = "global",
) -> float | dict[int, float]:
    r"""Isolation distance per Harris et al. (2000).

    For a cluster of size *n_c*, the isolation distance is the
    Mahalanobis radius from the cluster centroid that encloses
    *n_c* non-cluster spikes.  Larger values indicate better
    isolation.

    **Modes.**

    - ``mode="global"`` (default, legacy) — non-cluster spikes are
      pooled across *all* other clusters.  Matches the original
      Harris et al. (2000) and Schmitzer-Torbert et al. (2005)
      formulation.
    - ``mode="worst_pair"`` (modern best practice) — for each other
      cluster *B*, compute the isolation distance using only the
      spikes of *B* as the "out" set, then return the *minimum*
      across other clusters (the *worst* neighbour).  Use this when
      you want to flag clusters that overlap badly with a *specific*
      neighbour even if the global isolation distance is large.
      This is the convention recommended by Sibille et al. (2024,
      *eNeuro*) and the Allen Institute `ecephys_spike_sorting`
      pipeline.

    The cluster covariance is estimated with the **Ledoit–Wolf shrinkage
    estimator** (``sklearn.covariance.LedoitWolf``) rather than the
    sample covariance.  This avoids the silent eigenvalue truncation of
    ``np.linalg.pinv`` on small or near-collinear clusters.  See
    :func:`_ledoit_wolf_precision` for the rationale.

    Args:
        features: Feature matrix ``(n_samples, n_features)``.
        cluster_labels: Cluster label per sample.
        all_clusters: Return dict of per-cluster values (``True``)
            or a single cluster (``False``).
        cluster_id: Cluster ID when ``all_clusters=False``.
        mode: ``"global"`` (default, Harris 2000) or ``"worst_pair"``
            (modern best practice, Sibille 2024).

    Returns:
        Isolation distance (float) or ``{cluster_id: float}`` dict.
        ``np.nan`` when the cluster is too small or there are fewer
        non-cluster spikes than cluster spikes (under ``"global"``)
        / fewer spikes in the worst neighbour than in the target
        cluster (under ``"worst_pair"``).

    References:
        Harris, K. D., Henze, D. A., Csicsvari, J., Hirase, H. &
        Buzsáki, G. (2000).  *Accuracy of tetrode spike separation as
        determined by simultaneous intracellular and extracellular
        measurements*.  Journal of Neurophysiology 84(1), 401–414.
        doi:10.1152/jn.2000.84.1.401.

        Schmitzer-Torbert, N., Jackson, J., Henze, D., Harris, K. &
        Redish, A. D. (2005).  *Quantitative measures of cluster
        quality for use in extracellular recordings*.  Neuroscience
        131(1), 1–11.  doi:10.1016/j.neuroscience.2004.09.066.

        Sibille, J. et al. (2024).  *Assessing cross-contamination
        in spike-sorted electrophysiology data*.  eNeuro 11(8),
        ENEURO.0554-23.2024.  doi:10.1523/ENEURO.0554-23.2024.
    """
    if mode not in ("global", "worst_pair"):
        raise ValueError(f"mode must be 'global' or 'worst_pair', got {mode!r}.")
    _validate_cluster_args(all_clusters, cluster_id)
    cluster_labels = np.asarray(cluster_labels)
    unique = np.unique(cluster_labels)

    def _iso_global(cid: int) -> float:
        mask = cluster_labels == cid
        n_c = int(mask.sum())
        if n_c < 2:
            return np.nan
        f_in = features[mask]
        f_out = features[~mask]
        if len(f_out) < n_c:
            return np.nan
        cov_inv = _ledoit_wolf_precision(f_in)
        mean = f_in.mean(axis=0)
        d2 = _mahalanobis_sq(f_out, mean, cov_inv)
        d2_sorted = np.sort(d2)
        return float(d2_sorted[n_c - 1])

    def _iso_worst_pair(cid: int) -> float:
        # For each other cluster B, the isolation distance is the
        # Mahalanobis radius around A's centroid that encloses n_A
        # spikes of B.  Worst-case = minimum over all B (the
        # closest competing cluster).
        mask = cluster_labels == cid
        n_c = int(mask.sum())
        if n_c < 2:
            return np.nan
        f_in = features[mask]
        cov_inv = _ledoit_wolf_precision(f_in)
        mean = f_in.mean(axis=0)
        worst = np.inf
        for other in unique:
            if other == cid:
                continue
            f_other = features[cluster_labels == other]
            if len(f_other) < n_c:
                # Not enough spikes in B to enclose n_A — skip this
                # neighbour rather than degrading the worst-case
                # score on a small-cluster artefact.
                continue
            d2 = _mahalanobis_sq(f_other, mean, cov_inv)
            d2_sorted = np.sort(d2)
            worst = min(worst, float(d2_sorted[n_c - 1]))
        return worst if np.isfinite(worst) else np.nan

    _iso_one = _iso_worst_pair if mode == "worst_pair" else _iso_global

    if all_clusters:
        return {int(c): _iso_one(int(c)) for c in unique}
    return _iso_one(int(cluster_id))


# ---------------------------------------------------------------------------
# L-ratio
# ---------------------------------------------------------------------------


def l_ratio(
    features: npt.NDArray,
    cluster_labels: npt.NDArray,
    all_clusters: bool = True,
    cluster_id: int | None = None,
    *,
    mode: str = "global",
) -> float | dict[int, float]:
    r"""L-ratio: complement of isolation distance via chi-squared CDF.

    .. math::

        L = \frac{1}{n_c} \sum_{x \notin C} \bigl(1 - \chi^2_d\,
            \mathrm{cdf}(\,d_M^2(x, \mu_C, \Sigma_C)\,)\bigr)

    Smaller values (< 0.1) indicate better isolation.

    **Modes.**

    - ``mode="global"`` (default, legacy) — the sum runs over all
      non-cluster spikes.  Matches Schmitzer-Torbert et al. (2005).
    - ``mode="worst_pair"`` (modern best practice) — for each other
      cluster *B*, compute the L-ratio using only the spikes of *B*
      as the "out" set, then return the *maximum* across other
      clusters (the *worst* neighbour — high L-ratio = poor
      isolation).  Same motivation as in
      :func:`isolation_distance(mode="worst_pair")`.

    The cluster covariance is estimated with the **Ledoit–Wolf shrinkage
    estimator** (``sklearn.covariance.LedoitWolf``); see
    :func:`_ledoit_wolf_precision` and :func:`isolation_distance` for
    the rationale.

    Args:
        features: Feature matrix ``(n_samples, n_features)``.
        cluster_labels: Cluster label per sample.
        all_clusters: Return dict (``True``) or single value (``False``).
        cluster_id: Cluster ID when ``all_clusters=False``.
        mode: ``"global"`` (default, Schmitzer-Torbert 2005) or
            ``"worst_pair"`` (Sibille 2024).

    Returns:
        L-ratio (float) or ``{cluster_id: float}`` dict.

    References:
        Schmitzer-Torbert, N., Jackson, J., Henze, D., Harris, K. &
        Redish, A. D. (2005).  *Quantitative measures of cluster
        quality for use in extracellular recordings*.  Neuroscience
        131(1), 1–11.  doi:10.1016/j.neuroscience.2004.09.066.

        Sibille, J. et al. (2024).  *Assessing cross-contamination
        in spike-sorted electrophysiology data*.  eNeuro 11(8),
        ENEURO.0554-23.2024.  doi:10.1523/ENEURO.0554-23.2024.
    """
    if mode not in ("global", "worst_pair"):
        raise ValueError(f"mode must be 'global' or 'worst_pair', got {mode!r}.")
    _validate_cluster_args(all_clusters, cluster_id)
    cluster_labels = np.asarray(cluster_labels)
    df = features.shape[1]
    unique = np.unique(cluster_labels)

    def _lr_global(cid: int) -> float:
        mask = cluster_labels == cid
        n_c = int(mask.sum())
        if n_c < 2:
            return np.nan
        f_in = features[mask]
        f_out = features[~mask]
        if len(f_out) == 0:
            # Single-cluster recording — L-ratio is undefined, not "perfect"
            return np.nan
        cov_inv = _ledoit_wolf_precision(f_in)
        mean = f_in.mean(axis=0)
        d2 = _mahalanobis_sq(f_out, mean, cov_inv)
        L = float(np.sum(1.0 - sp_stats.chi2.cdf(d2, df=df)))
        return L / n_c

    def _lr_worst_pair(cid: int) -> float:
        mask = cluster_labels == cid
        n_c = int(mask.sum())
        if n_c < 2:
            return np.nan
        f_in = features[mask]
        cov_inv = _ledoit_wolf_precision(f_in)
        mean = f_in.mean(axis=0)
        worst = -np.inf
        any_other = False
        for other in unique:
            if other == cid:
                continue
            f_other = features[cluster_labels == other]
            if len(f_other) == 0:
                continue
            any_other = True
            d2 = _mahalanobis_sq(f_other, mean, cov_inv)
            L = float(np.sum(1.0 - sp_stats.chi2.cdf(d2, df=df))) / n_c
            worst = max(worst, L)
        if not any_other:
            return np.nan
        return worst

    _lr_one = _lr_worst_pair if mode == "worst_pair" else _lr_global

    if all_clusters:
        return {int(c): _lr_one(int(c)) for c in unique}
    return _lr_one(int(cluster_id))


# ---------------------------------------------------------------------------
# d-prime (signal detection theory)
# ---------------------------------------------------------------------------


def _cluster_mean_per_dim_variance(
    features: npt.NDArray,
    cluster_labels: npt.NDArray,
    unique: npt.NDArray,
) -> dict[int, tuple[npt.NDArray, float]]:
    """Per-cluster mean vector and *mean per-feature* variance.

    The variance returned is ``mean(var(Fc, axis=0))`` — the average
    variance across feature dimensions, **not** the global variance of
    the flattened cluster matrix.  These two are equivalent only when
    the per-feature means are identical, which is essentially never the
    case for waveform-shaped feature matrices.

    Returns ``{cluster_id: (mean_vector, sigma_squared)}``.  Clusters
    with fewer than two samples receive ``sigma_squared = nan`` so the
    caller can decide how to handle them (typically: skip the pair and
    propagate ``nan``).
    """
    stats: dict[int, tuple[npt.NDArray, float]] = {}
    for c in unique:
        Fc = features[cluster_labels == c]
        if len(Fc) < 2:
            stats[int(c)] = (Fc.mean(axis=0), np.nan)
        else:
            stats[int(c)] = (
                Fc.mean(axis=0),
                float(np.mean(np.var(Fc, axis=0))),
            )
    return stats


def d_prime_pairwise_matrix(
    features: npt.NDArray,
    cluster_labels: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Full pairwise d-prime matrix between every pair of clusters.

    ``d'(A, B) = |μ_A − μ_B|₂ / sqrt(0.5 (σ²_A + σ²_B))``

    where σ² is the average per-feature variance of the cluster
    (``mean(var(Fc, axis=0))``).  This is the formulation referenced in
    the spike-sorting literature; using ``np.var(Fc)`` (global variance
    of the flattened matrix) is mathematically incorrect because it
    folds the between-feature mean spread into the noise term.

    Args:
        features: Feature matrix ``(n_samples, n_features)``.
        cluster_labels: Cluster label per sample.

    Returns:
        ``(d_matrix, cluster_ids)`` where *d_matrix* is a symmetric
        ``(n_clusters, n_clusters)`` array with the diagonal set to
        ``nan``, and *cluster_ids* is the sorted unique label array.
        Off-diagonal entries are ``nan`` whenever a cluster has fewer
        than two samples or the pooled standard deviation falls below
        the float-rounding floor (``< 1e-12``).  The epsilon guard
        replaces an exact ``== 0`` check so ``np.sqrt(tiny + tiny)``
        rounding doesn't slip through and divide by a near-zero value.
    """
    cluster_labels = np.asarray(cluster_labels)
    unique = np.sort(np.unique(cluster_labels))
    n = len(unique)
    stats = _cluster_mean_per_dim_variance(features, cluster_labels, unique)

    mat = np.full((n, n), np.nan, dtype=np.float64)
    for i, ci in enumerate(unique):
        mu_a, var_a = stats[int(ci)]
        for j, cj in enumerate(unique):
            if i == j:
                continue
            mu_b, var_b = stats[int(cj)]
            if np.isnan(var_a) or np.isnan(var_b):
                continue
            pooled_std = np.sqrt(0.5 * (var_a + var_b))
            # Use an absolute floor rather than exact equality so float
            # rounding from `np.sqrt(tiny + tiny)` doesn't slip through
            # and produce a division-by-near-zero blow-up below.
            if pooled_std < 1e-12:
                continue
            mat[i, j] = float(np.linalg.norm(mu_a - mu_b) / pooled_std)
    return mat, unique


def d_prime(
    features: npt.NDArray,
    cluster_labels: npt.NDArray,
    all_clusters: bool = True,
    cluster_id: int | None = None,
) -> float | dict[int, float]:
    """d-prime: cluster separation in feature space.

    For each cluster, d′ is the minimum pairwise separation across all
    other clusters:

    ``d' = |μ_A − μ_B|₂ / sqrt(0.5 (σ²_A + σ²_B))``

    where σ² is the average per-feature variance of the cluster
    (``mean(var(Fc, axis=0))``).  Larger values indicate better
    separation (> 2 is good, > 4 is excellent).

    Args:
        features: Feature matrix ``(n_samples, n_features)``.
        cluster_labels: Cluster label per sample.
        all_clusters: Return dict (``True``) or single value (``False``).
        cluster_id: Cluster ID when ``all_clusters=False``.

    Returns:
        d-prime (float) or ``{cluster_id: float}`` dict.  Returns
        ``np.nan`` for any cluster where the minimum cannot be computed
        (single cluster, undersized cluster, or zero pooled std).
    """
    _validate_cluster_args(all_clusters, cluster_id)
    cluster_labels = np.asarray(cluster_labels)
    mat, unique = d_prime_pairwise_matrix(features, cluster_labels)
    if len(unique) < 2:
        result = {int(c): np.nan for c in unique}
        return result if all_clusters else result[int(cluster_id)]

    per_cluster: dict[int, float] = {}
    for i, c in enumerate(unique):
        row = mat[i]
        # Drop the diagonal NaN; remaining NaNs come from invalid pairs
        row_valid = row[~np.isnan(row)]
        per_cluster[int(c)] = float(row_valid.min()) if row_valid.size else np.nan

    if all_clusters:
        return per_cluster
    return per_cluster[int(cluster_id)]


# ---------------------------------------------------------------------------
# Peak amplitude SNR
# ---------------------------------------------------------------------------


def peak_amplitude_snr(
    waveforms: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    all_clusters: bool = True,
    cluster_id: int | None = None,
    baseline_frac: float = 0.1,
) -> float | dict[int, float]:
    """Peak amplitude signal-to-noise ratio.

    ``SNR = max|mean_waveform| / std(baseline)``

    where *baseline* is the first *baseline_frac* fraction of the
    waveform snippet (before the spike peak).

    Args:
        waveforms: Waveform matrix ``(n_spikes, snippet_length)``.
        cluster_labels: Cluster labels.  Required when ``all_clusters=True``
            for per-cluster results.
        all_clusters: Per-cluster dict (``True``) or single cluster.
        cluster_id: Cluster ID when ``all_clusters=False``.
        baseline_frac: Fraction of snippet used as baseline (0–1).

    Returns:
        SNR (float) or ``{cluster_id: float}`` dict.
    """
    _validate_cluster_args(all_clusters, cluster_id)

    baseline_n = max(1, int(waveforms.shape[1] * baseline_frac))

    def _snr_one(w: npt.NDArray) -> float:
        if len(w) < 2:
            return np.nan
        mean_wv = np.mean(w, axis=0, dtype=np.float64)
        signal = float(np.max(np.abs(mean_wv)))
        noise = float(np.std(w[:, :baseline_n]))
        if noise == 0:
            return np.nan
        return signal / noise

    if cluster_labels is None:
        return _snr_one(waveforms)

    cluster_labels = np.asarray(cluster_labels)
    if all_clusters:
        return {int(c): _snr_one(waveforms[cluster_labels == c]) for c in np.unique(cluster_labels)}
    return _snr_one(waveforms[cluster_labels == cluster_id])


# ---------------------------------------------------------------------------
# Waveform stability over time
# ---------------------------------------------------------------------------


def waveform_stability(
    spike_times: npt.NDArray,
    waveforms: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    all_clusters: bool = True,
    cluster_id: int | None = None,
    percentiles: Sequence[float] = (25, 75),
    first_last_only: bool = True,
) -> float | dict[int, float]:
    """Waveform shape stability over time.

    Sort spikes chronologically, split into percentile windows, and
    compute Pearson correlation between the mean waveforms of the
    early and late windows.

    Args:
        spike_times: Spike times (for temporal ordering).
        waveforms: Waveform matrix ``(n_spikes, snippet_length)``.
        cluster_labels: Cluster labels.
        all_clusters: Per-cluster dict (``True``) or single cluster.
        cluster_id: Cluster ID when ``all_clusters=False``.
        percentiles: Time percentiles defining windows.
            Default ``(25, 75)`` compares first quartile vs last.
        first_last_only: If ``True`` (default), only compare the
            first and last percentile.  If ``False``, return the
            minimum correlation across all consecutive pairs.

    Returns:
        Pearson *r* (float, 0–1).  1.0 = perfect stability.
        ``np.nan`` if insufficient spikes in any window.
    """
    _validate_cluster_args(all_clusters, cluster_id)

    def _stab_one(times: npt.NDArray, waves: npt.NDArray) -> float:
        if len(times) < 4:
            return np.nan
        order = np.argsort(times)
        waves_sorted = waves[order]
        pct_vals = np.percentile(times[order], percentiles)
        times_sorted = times[order]

        # Build window boundaries
        edges = [times_sorted[0]] + list(pct_vals) + [times_sorted[-1] + 1e-12]
        means = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (times_sorted >= lo) & (times_sorted < hi)
            if mask.sum() < 2:
                return np.nan
            means.append(np.mean(waves_sorted[mask], axis=0, dtype=np.float64))

        if first_last_only:
            r, _ = sp_stats.pearsonr(means[0], means[-1])
            return float(r)
        # Min correlation across consecutive pairs
        r_min = 1.0
        for i in range(len(means) - 1):
            r, _ = sp_stats.pearsonr(means[i], means[i + 1])
            r_min = min(r_min, float(r))
        return r_min

    if cluster_labels is None:
        return _stab_one(spike_times, waveforms)

    cluster_labels = np.asarray(cluster_labels)
    if all_clusters:
        result = {}
        for c in np.unique(cluster_labels):
            mask = cluster_labels == c
            result[int(c)] = _stab_one(spike_times[mask], waveforms[mask])
        return result
    mask = cluster_labels == cluster_id
    return _stab_one(spike_times[mask], waveforms[mask])


# ---------------------------------------------------------------------------
# Amplitude drift (Spearman correlation of amplitude vs. time)
# ---------------------------------------------------------------------------


def amplitude_drift(
    waveforms: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    all_clusters: bool = True,
    cluster_id: int | None = None,
) -> float | dict[int, float]:
    """Spearman correlation of spike peak amplitude vs. spike index.

    Detects systematic amplitude changes over the recording.
    ``|r| > 0.3`` suggests significant drift.

    Args:
        waveforms: Waveform matrix ``(n_spikes, snippet_length)``.
        cluster_labels: Cluster labels.
        all_clusters: Per-cluster dict or single value.
        cluster_id: Cluster ID when ``all_clusters=False``.

    Returns:
        Spearman *r* (float, −1 to 1) or ``{cluster_id: float}`` dict.
        ``np.nan`` if fewer than 3 spikes.
    """
    _validate_cluster_args(all_clusters, cluster_id)

    def _drift_one(w: npt.NDArray) -> float:
        if len(w) < 3:
            return np.nan
        amps = np.max(w, axis=1) - np.min(w, axis=1)
        r, _ = sp_stats.spearmanr(np.arange(len(amps)), amps)
        return float(r)

    if cluster_labels is None:
        return _drift_one(waveforms)

    cluster_labels = np.asarray(cluster_labels)
    if all_clusters:
        return {
            int(c): _drift_one(waveforms[cluster_labels == c]) for c in np.unique(cluster_labels)
        }
    return _drift_one(waveforms[cluster_labels == cluster_id])


# ---------------------------------------------------------------------------
# Fraction of missing spikes
# ---------------------------------------------------------------------------


def fraction_missing(
    waveforms: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    all_clusters: bool = True,
    cluster_id: int | None = None,
    normality_warn: bool = True,
    method: str = "gaussian",
) -> float | dict[int, float]:
    r"""Estimate fraction of undetected spikes from amplitude distribution.

    Estimates the fraction of the underlying spike-amplitude
    distribution that sits below the lowest observed amplitude — i.e.
    the spikes the threshold-based detector would have missed.  This
    is the standard "amplitude cutoff" / "missing-spikes" metric used
    in the spike-sorting QC literature (Hill et al. 2011, the Allen
    Institute *ecephys_spike_sorting* pipeline).

    **Methods.**

    - ``method="gaussian"`` (default, legacy) — fit a normal to the
      peak-amplitude histogram and report ``Φ((threshold − μ) / σ)``.
      The Hill 2011 / Allen Institute convention.  Assumes the
      *true* spike-amplitude distribution is Gaussian; usually wrong
      for real recordings (V1 amplitude distributions tend to be
      lognormal — Buzsáki & Mizuseki 2014) but is the universally
      reported number.  Use this for cross-paper comparability.
    - ``method="lognormal"`` — fit on the *log-amplitudes* and
      compute the tail probability there.  Better-calibrated for the
      observed shape of V1 amplitude distributions; reports
      systematically *higher* missing fractions than the Gaussian
      method for skewed clusters (because the Gaussian tail
      underweights the low-amplitude side).  Use this when you've
      visually inspected the amplitude histogram and confirmed it is
      lognormal-shaped (or when the ``normality_warn`` KS test
      rejects normality strongly).
    - ``method="empirical"`` — non-parametric tail estimate based on
      the empirical CDF and a Gaussian kernel extrapolation
      (Silverman's rule).  Makes no parametric assumption about the
      shape; the trade-off is higher variance for small clusters.
      Best for highly non-Gaussian / multi-modal distributions
      where neither Gaussian nor lognormal fits are appropriate.

    .. warning::

       For **all three** methods, multi-modal amplitude distributions
       (mixed units, drift across the recording, bursts) violate the
       single-distribution assumption and the returned number is
       meaningless.  The KS-test warning (``normality_warn=True``,
       applies only to ``method="gaussian"``) is a cheap diagnostic
       but is not exhaustive — inspect the amplitude histogram
       directly before trusting any of these estimates for a low-SNR
       cluster.

    Args:
        waveforms: Waveform matrix ``(n_spikes, snippet_length)``.
        cluster_labels: Cluster labels.
        all_clusters: Per-cluster dict or single value.
        cluster_id: Cluster ID when ``all_clusters=False``.
        normality_warn: If ``True`` (default), warn when the KS test
            against a fitted normal rejects normality at *p < 0.01*.
            Only consulted for ``method="gaussian"``.
        method: ``"gaussian"`` (default, legacy / Hill 2011),
            ``"lognormal"``, or ``"empirical"``.

    Returns:
        Fraction missing (float, 0–1) or ``{cluster_id: float}`` dict.
        ``np.nan`` if fewer than 10 spikes (or 20 for the empirical
        method, which needs more data to estimate the tail).

    Raises:
        ValueError: If *method* is not one of the supported strings.

    References:
        Hill, D. N., Mehta, S. B. & Kleinfeld, D. (2011).  *Quality
        metrics to accompany spike sorting of extracellular signals*.
        Journal of Neuroscience 31(24), 8699–8705.
        doi:10.1523/JNEUROSCI.0971-11.2011.

        Buzsáki, G. & Mizuseki, K. (2014).  *The log-dynamic brain:
        how skewed distributions affect network operations*.  Nature
        Reviews Neuroscience 15(4), 264–278.  doi:10.1038/nrn3687.

        Silverman, B. W. (1986).  *Density Estimation for Statistics
        and Data Analysis*.  Chapman & Hall, London.  (For
        kernel-bandwidth selection used by ``method="empirical"``.)

        Allen Institute *ecephys_spike_sorting* — `amplitude_cutoff`
        reference implementation:
        https://github.com/AllenInstitute/ecephys_spike_sorting/
    """
    if method not in ("gaussian", "lognormal", "empirical"):
        raise ValueError(
            f"method must be one of 'gaussian', 'lognormal', 'empirical'; got {method!r}."
        )
    _validate_cluster_args(all_clusters, cluster_id)

    def _frac_one(w: npt.NDArray, label: object | None = None) -> float:
        # Sample-size floor: 10 spikes for parametric fits (the legacy
        # Gaussian and the lognormal); 20 for the empirical/KDE tail
        # which is variance-dominated for tiny samples.
        n_min = 20 if method == "empirical" else 10
        if len(w) < n_min:
            return np.nan
        amps = np.max(w, axis=1) - np.min(w, axis=1)
        if method == "gaussian":
            mu, sigma = sp_stats.norm.fit(amps)
            if sigma == 0:
                return np.nan
            if normality_warn and len(amps) >= 20:
                _ks, ks_p = sp_stats.kstest(amps, "norm", args=(mu, sigma))
                if ks_p < 0.01:
                    tag = f" (cluster {label})" if label is not None else ""
                    warnings.warn(
                        f"fraction_missing{tag}: amplitude distribution is "
                        f"not normal (KS p={ks_p:.2g}); the Gaussian-tail "
                        f"estimate may be misleading. Consider "
                        f"method='lognormal' or method='empirical', or "
                        f"inspect the amplitude histogram.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
            threshold = amps.min()
            return float(sp_stats.norm.cdf(threshold, loc=mu, scale=sigma))

        if method == "lognormal":
            # Strictly positive amplitudes are required for the log
            # transform.  Real peak-to-peak waveform amplitudes are
            # always > 0; this guard catches edge cases where the
            # waveform is identically zero (which would also give
            # NaN under the Gaussian path via sigma == 0).
            if np.any(amps <= 0):
                return np.nan
            log_amps = np.log(amps)
            mu_log = float(np.mean(log_amps))
            sigma_log = float(np.std(log_amps, ddof=1))
            if sigma_log == 0:
                return np.nan
            threshold_log = float(np.log(amps.min()))
            return float(sp_stats.norm.cdf(threshold_log, loc=mu_log, scale=sigma_log))

        # method == "empirical"
        # Estimate the underlying density with a Gaussian kernel (KDE)
        # at Silverman's bandwidth and integrate the tail below the
        # observed minimum.  This is the non-parametric analogue of
        # the Gaussian-tail estimator and makes no assumption about
        # the shape of the amplitude distribution.
        amps_sorted = np.sort(amps)
        # Silverman's rule of thumb: h = 1.06 σ̂ n^(-1/5).
        n = len(amps_sorted)
        sigma_est = float(np.std(amps_sorted, ddof=1))
        if sigma_est == 0:
            return np.nan
        h = 1.06 * sigma_est * n ** (-1.0 / 5.0)
        # Tail probability under the KDE: ∫_{-∞}^{x_min} f̂(t) dt
        # = (1/n) Σ Φ((x_min − amp_i) / h).
        x_min = float(amps_sorted[0])
        tail = float(np.mean(sp_stats.norm.cdf((x_min - amps_sorted) / h)))
        # Clamp to [0, 0.5] — the empirical method can in principle
        # produce a tiny number > 0.5 from kernel smoothing of a
        # near-symmetric distribution; saturate consistently with
        # the Gaussian / lognormal paths.
        return max(0.0, min(0.5, tail))

    if cluster_labels is None:
        return _frac_one(waveforms)

    cluster_labels = np.asarray(cluster_labels)
    if all_clusters:
        return {
            int(c): _frac_one(waveforms[cluster_labels == c], label=int(c))
            for c in np.unique(cluster_labels)
        }
    return _frac_one(
        waveforms[cluster_labels == cluster_id],
        label=cluster_id,
    )
