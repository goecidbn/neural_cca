"""Spike-sorting quality metrics.

Functions for evaluating spike-sorting quality: silhouette-based
misclassification, refractory-period violations, signal-to-noise
ratio, pre-stimulus spike counts, isolation distance, L-ratio,
d-prime, waveform stability, amplitude drift, and missing spikes.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Optional

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
    cluster_id: Optional[int] = None,
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
        deviation is zero (degenerate input — single waveform or
        identical traces).
    """
    W_bar = np.mean(waveforms, axis=0, dtype=np.float64)
    sig_amp = np.max(W_bar) - np.min(W_bar)
    noise = waveforms - W_bar
    noise_std = float(np.std(noise))
    if noise_std == 0:
        return np.nan
    return float(sig_amp / (2.0 * noise_std))


def calc_weighted_snr(
    waveforms: npt.NDArray,
    cluster_labels: npt.NDArray,
) -> float:
    """Cluster-size-weighted SNR across all clusters.

    For each cluster, computes ``est_snr`` and weights it by the
    proportion of snippets belonging to that cluster.

    Args:
        waveforms: Waveform matrix, shape ``(n_snippets, snippet_length)``.
        cluster_labels: Cluster label per snippet.

    Returns:
        Weighted SNR (float).
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    weights = counts / np.sum(counts)

    weighted_snr = 0.0
    for cid, weight in zip(unique, weights):
        snr = est_snr(waveforms[cluster_labels == cid])
        weighted_snr += snr * weight

    return float(weighted_snr)


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
        ValueError: On invalid argument combinations.
    """
    _validate_cluster_args(all_clusters, cluster_id)
    if not all_clusters and cluster_labels is None:
        raise ValueError(
            "cluster_labels must be provided when all_clusters is False."
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
# Helper: validate all_clusters / cluster_id combination
# ---------------------------------------------------------------------------

def _validate_cluster_args(
    all_clusters: bool, cluster_id: int | None,
) -> None:
    if all_clusters and cluster_id is not None:
        raise ValueError(
            "Cannot specify 'cluster_id' when 'all_clusters' is True."
        )
    if not all_clusters and cluster_id is None:
        raise ValueError(
            "Must specify 'cluster_id' when 'all_clusters' is False."
        )


def _mahalanobis_sq(
    features: npt.NDArray, mean: npt.NDArray, cov_inv: npt.NDArray,
) -> npt.NDArray:
    """Squared Mahalanobis distance of each row in *features* from *mean*."""
    diff = features - mean
    return np.sum(diff @ cov_inv * diff, axis=1)


def _ledoit_wolf_precision(features: npt.NDArray) -> npt.NDArray:
    """Ledoit–Wolf shrunk covariance, returned as its inverse (precision).

    The Ledoit–Wolf estimator linearly shrinks the sample covariance
    toward a scaled identity, with the shrinkage intensity chosen
    analytically to minimise expected MSE.  This is the standard fix
    for ill-conditioned sample covariances when the number of samples
    is comparable to or smaller than the feature dimension — exactly
    the regime in which spike-sorting feature matrices live (clusters
    of a few hundred spikes in a 12–32 dimensional PCA space).

    Replaces ``np.linalg.pinv(np.cov(X, rowvar=False))``, which silently
    discards eigenvalues below the pinv tolerance and produces unstable
    Mahalanobis distances on small or near-degenerate clusters.

    References:
        Ledoit, O. & Wolf, M. (2004).  *A well-conditioned estimator
        for large-dimensional covariance matrices*.  Journal of
        Multivariate Analysis, 88(2), 365–411.
    """
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
) -> float | dict[int, float]:
    """Isolation distance per Harris et al. (2000).

    For a cluster of size *n_c*, compute the Mahalanobis distance from
    the cluster centroid at which *n_c* non-cluster spikes are enclosed.
    Larger values indicate better isolation.

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

    Returns:
        Isolation distance (float) or ``{cluster_id: float}`` dict.
        ``np.nan`` when the cluster is too small or there are fewer
        non-cluster spikes than cluster spikes.
    """
    _validate_cluster_args(all_clusters, cluster_id)
    cluster_labels = np.asarray(cluster_labels)

    def _iso_one(cid: int) -> float:
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

    if all_clusters:
        return {int(c): _iso_one(c) for c in np.unique(cluster_labels)}
    return _iso_one(cluster_id)


# ---------------------------------------------------------------------------
# L-ratio
# ---------------------------------------------------------------------------

def l_ratio(
    features: npt.NDArray,
    cluster_labels: npt.NDArray,
    all_clusters: bool = True,
    cluster_id: int | None = None,
) -> float | dict[int, float]:
    """L-ratio: complement of isolation distance via chi-squared CDF.

    ``L = sum(1 - chi2.cdf(D², df)) / n_cluster`` for all non-cluster
    spikes.  Smaller values (< 0.1) indicate better isolation.

    The cluster covariance is estimated with the **Ledoit–Wolf shrinkage
    estimator** (``sklearn.covariance.LedoitWolf``); see
    :func:`_ledoit_wolf_precision` and :func:`isolation_distance` for
    the rationale.

    Args:
        features: Feature matrix ``(n_samples, n_features)``.
        cluster_labels: Cluster label per sample.
        all_clusters: Return dict (``True``) or single value (``False``).
        cluster_id: Cluster ID when ``all_clusters=False``.

    Returns:
        L-ratio (float) or ``{cluster_id: float}`` dict.
    """
    _validate_cluster_args(all_clusters, cluster_id)
    cluster_labels = np.asarray(cluster_labels)
    df = features.shape[1]

    def _lr_one(cid: int) -> float:
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

    if all_clusters:
        return {int(c): _lr_one(c) for c in np.unique(cluster_labels)}
    return _lr_one(cluster_id)


# ---------------------------------------------------------------------------
# d-prime (signal detection theory)
# ---------------------------------------------------------------------------

def _cluster_mean_per_dim_variance(
    features: npt.NDArray, cluster_labels: npt.NDArray, unique: npt.NDArray,
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
        than two samples or the pooled standard deviation is zero.
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
            if pooled_std == 0:
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
    if not all_clusters and cluster_id is None:
        raise ValueError(
            "Must specify 'cluster_id' when 'all_clusters' is False."
        )

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
        return {
            int(c): _snr_one(waveforms[cluster_labels == c])
            for c in np.unique(cluster_labels)
        }
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
    if not all_clusters and cluster_id is None:
        raise ValueError(
            "Must specify 'cluster_id' when 'all_clusters' is False."
        )

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
    if not all_clusters and cluster_id is None:
        raise ValueError(
            "Must specify 'cluster_id' when 'all_clusters' is False."
        )

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
            int(c): _drift_one(waveforms[cluster_labels == c])
            for c in np.unique(cluster_labels)
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
) -> float | dict[int, float]:
    """Estimate fraction of undetected spikes from amplitude distribution.

    Fit a Gaussian to the peak-amplitude histogram and estimate the
    fraction of the distribution below the observed minimum amplitude.

    .. warning::
       This estimator assumes the underlying amplitude distribution is
       approximately Gaussian.  For multi-modal distributions (mixed
       units, contamination, drift) the result is meaningless.  When
       *normality_warn* is ``True`` (the default) a Kolmogorov–Smirnov
       test is run against the fitted normal and a warning is emitted
       when ``KS p < 0.01``.  Inspect the amplitude histogram before
       trusting this number — see also the :doc:`known issues
       <../known_issues>` page for alternatives (Gaussian-mixture fits,
       lognormal fits, empirical-CDF tail estimation).

    Args:
        waveforms: Waveform matrix ``(n_spikes, snippet_length)``.
        cluster_labels: Cluster labels.
        all_clusters: Per-cluster dict or single value.
        cluster_id: Cluster ID when ``all_clusters=False``.
        normality_warn: If ``True`` (default), warn when the KS test
            against a fitted normal rejects normality at *p < 0.01*.

    Returns:
        Fraction missing (float, 0–1) or ``{cluster_id: float}`` dict.
        ``np.nan`` if fewer than 10 spikes.
    """
    if not all_clusters and cluster_id is None:
        raise ValueError(
            "Must specify 'cluster_id' when 'all_clusters' is False."
        )

    def _frac_one(w: npt.NDArray, label: object | None = None) -> float:
        if len(w) < 10:
            return np.nan
        amps = np.max(w, axis=1) - np.min(w, axis=1)
        mu, sigma = sp_stats.norm.fit(amps)
        if sigma == 0:
            # All amplitudes identical — Gaussian-tail estimator undefined
            return np.nan
        if normality_warn and len(amps) >= 20:
            # KS test against the fitted normal — cheap and standard
            _ks, ks_p = sp_stats.kstest(amps, "norm", args=(mu, sigma))
            if ks_p < 0.01:
                tag = f" (cluster {label})" if label is not None else ""
                warnings.warn(
                    f"fraction_missing{tag}: amplitude distribution is "
                    f"not normal (KS p={ks_p:.2g}); the Gaussian-tail "
                    f"estimate may be misleading. Consider inspecting "
                    f"the amplitude histogram or using an alternative "
                    f"estimator.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        threshold = amps.min()
        return float(sp_stats.norm.cdf(threshold, loc=mu, scale=sigma))

    if cluster_labels is None:
        return _frac_one(waveforms)

    cluster_labels = np.asarray(cluster_labels)
    if all_clusters:
        return {
            int(c): _frac_one(waveforms[cluster_labels == c], label=int(c))
            for c in np.unique(cluster_labels)
        }
    return _frac_one(
        waveforms[cluster_labels == cluster_id], label=cluster_id,
    )
