"""Spike sorting pipeline — clustering, evaluation, orchestration.

End-to-end workflow: cluster waveforms, select optimal *k*, evaluate
sorting quality, compute orientation-selectivity metrics per cluster,
and produce a diagnostic summary figure.
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .containers import SortingData, SortingResult
from .metrics import (
    amplitude_drift,
    calc_weighted_snr,
    d_prime,
    est_snr,
    fraction_missing,
    isolation_distance,
    l_ratio,
    neg_silhouette_score,
    peak_amplitude_snr,
    rpvs,
    waveform_stability,
)
from .plotting import plot_sorting_summary, plot_k_search

# Cross-package import for orientation-selectivity metrics.  This is
# only used by ``evaluate_os_per_cluster``; if ``tuning`` is
# unavailable for any reason at install time the import will fail at
# module load and the user gets a clean error.  ``tuning`` is
# part of the same distribution so this is always installed.
from ..tuning.tuning import get_os_metrics

__all__ = [
    "SortingResult",
    "find_optimal_k",
    "sort_spikes",
    "evaluate_sorting",
    "evaluate_os_per_cluster",
    "run_sorting_pipeline",
    "PreprocessMode",
    "RngLike",
]


PreprocessMode = Literal["none", "center", "zscore", "pca", "zscore_pca"]
RngLike = "np.random.Generator | int | None"


def _as_seed(rng: np.random.Generator | int | None) -> int | None:
    """Coerce an rng spec to an integer seed for sklearn estimators.

    sklearn estimators accept ``None``, ``int``, or ``RandomState`` for
    their ``random_state`` argument but not :class:`numpy.random.Generator`.
    When the user passes a ``Generator``, we sample one int from it so the
    estimator gets a deterministic seed *derived from* the generator.
    Repeated calls with the same Generator therefore produce different
    sklearn seeds — this matches the usual expectation that a Generator
    is consumed as you draw from it.
    """
    if rng is None:
        return None
    if isinstance(rng, (int, np.integer)):
        return int(rng)
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(0, 2**31 - 1))
    raise TypeError(
        f"rng must be a Generator, int, or None; got {type(rng).__name__}"
    )


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def _center(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Subtract the per-feature mean."""
    return x - x.mean(axis=0, keepdims=True)


def _zscore(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Centre + scale to unit variance per feature.

    Zero-variance columns are left at zero rather than producing
    NaN / Inf, so a constant column does not blow up downstream
    sklearn estimators.
    """
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1.0)
    return (x - mu) / sigma


def _pca(
    x: npt.NDArray[np.float64],
    n_components: int | float | None,
    seed: int | None,
) -> npt.NDArray[np.float64]:
    """Fit PCA and return the projected scores.

    *n_components* defaults to ``0.95`` (keep enough components to
    explain 95 % of the variance) when ``None``.
    """
    n = 0.95 if n_components is None else n_components
    return PCA(n_components=n, random_state=seed).fit_transform(x)


def _preprocess_waveforms(
    waveforms: npt.NDArray[np.float64],
    preprocess: PreprocessMode,
    pca_components: int | float | None = None,
    rng: np.random.Generator | int | None = None,
) -> npt.NDArray[np.float64]:
    """Transform raw waveforms into the feature space used for clustering.

    The recommended pipeline for spike sorting is the chained
    ``"zscore_pca"`` mode: per-feature z-score so a single noisy
    sample cannot dominate the principal axes, then PCA to reduce
    dimensionality and decorrelate the features.  Plain ``"none"``
    feeds raw waveforms into KMeans, which is convenient for
    debugging but hard to defend in a methods section.

    Args:
        waveforms: ``(n_spikes, snippet_length)``.
        preprocess: One of:

            * ``"none"`` — return as-is.  KMeans sees raw waveforms.
              Convenient for debugging; not recommended for analysis.
            * ``"center"`` — subtract the per-feature mean.
            * ``"zscore"`` — centre and scale to unit variance per
              feature.  Zero-variance columns are left at zero rather
              than producing NaN / Inf.
            * ``"pca"`` — fit a PCA on centred waveforms and return
              the projected scores.  Centres internally but does not
              standardise variance, so a noisy sample can still
              dominate the principal axes.
            * ``"zscore_pca"`` — z-score followed by PCA.  This is the
              canonical "Z-score → PCA → KMeans" pipeline that gets
              used in spike-sorting methods sections.

        pca_components: Component count or variance ratio passed to
            :class:`sklearn.decomposition.PCA`.  Defaults to ``0.95``
            (keep enough components to explain 95 % of the variance).
            Ignored unless *preprocess* contains a PCA stage.
        rng: Generator, int seed, or ``None`` for PCA's randomised SVD.

    Returns:
        Feature matrix, ``(n_spikes, n_features)``.

    Raises:
        ValueError: If *preprocess* is not one of the supported modes.
    """
    seed = _as_seed(rng)

    if preprocess == "none":
        return waveforms
    if preprocess == "center":
        return _center(waveforms)
    if preprocess == "zscore":
        return _zscore(waveforms)
    if preprocess == "pca":
        return _pca(waveforms, pca_components, seed)
    if preprocess == "zscore_pca":
        return _pca(_zscore(waveforms), pca_components, seed)
    raise ValueError(
        f"Unknown preprocess mode: {preprocess!r}. "
        "Expected one of 'none', 'center', 'zscore', 'pca', 'zscore_pca'."
    )


def find_optimal_k(
    waveforms: npt.NDArray[np.float64],
    k_range: Sequence[int] = range(2, 8),
    rng: np.random.Generator | int | None = None,
    n_init: str | int = "auto",
    preprocess: PreprocessMode = "zscore_pca",
    pca_components: int | float | None = None,
) -> tuple[int, dict[int, float]]:
    """Select the best number of clusters via silhouette score.

    Args:
        waveforms: ``(n_spikes, snippet_length)``.
        k_range: Candidate cluster counts to evaluate.
        rng: Generator, int seed, or ``None`` for KMeans reproducibility.
        n_init: Number of KMeans initialisations.
        preprocess: Feature transformation applied before clustering.
            See ``_preprocess_waveforms`` for the supported modes.
        pca_components: Component count or variance ratio for the PCA
            mode (ignored otherwise).

    Returns:
        ``(best_k, scores)`` where *scores* maps each k to its mean
        silhouette score, both computed in the preprocessed feature
        space.
    """
    seed = _as_seed(rng)
    feats = _preprocess_waveforms(
        waveforms, preprocess,
        pca_components=pca_components, rng=seed,
    )
    scores: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(feats)
        scores[k] = float(silhouette_score(feats, labels))

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best_k, scores


def sort_spikes(
    waveforms: npt.NDArray[np.float64],
    n_clusters: int,
    rng: np.random.Generator | int | None = None,
    n_init: str | int = "auto",
    preprocess: PreprocessMode = "zscore_pca",
    pca_components: int | float | None = None,
) -> tuple[npt.NDArray[np.int64], KMeans]:
    """Run KMeans clustering on waveform snippets.

    Args:
        waveforms: ``(n_spikes, snippet_length)``.
        n_clusters: Number of clusters.
        rng: Generator, int seed, or ``None`` for KMeans reproducibility.
        n_init: Number of initialisations.
        preprocess: Feature transformation applied before clustering.
            See ``_preprocess_waveforms`` for the supported modes.
        pca_components: Component count or variance ratio for the PCA
            mode (ignored otherwise).

    Returns:
        ``(cluster_labels, fitted_kmeans_model)``.  The KMeans model is
        fit in the preprocessed feature space — its cluster centres live
        in that space, not the raw waveform space.
    """
    seed = _as_seed(rng)
    feats = _preprocess_waveforms(
        waveforms, preprocess,
        pca_components=pca_components, rng=seed,
    )
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    cluster_labels = km.fit_predict(feats).astype(np.int64)
    return cluster_labels, km


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------

def evaluate_sorting(
    data: SortingData,
    cluster_labels: npt.NDArray[np.int64],
    refractory_period: float = 0.001,
    compute_advanced: bool = True,
) -> dict:
    """Compute sorting-quality metrics.

    Core metrics (always computed):
        ``neg_silhouette_rel`` — fraction with negative silhouette.
        ``silhouette_mean`` — mean silhouette score.
        ``abs_rpvs`` — absolute refractory-period violations.
        ``rel_rpvs`` — RPV fraction.
        ``snr_weighted`` — cluster-size-weighted SNR.
        ``snr_per_cluster`` — per-cluster SNR dict.

    Advanced metrics (when *compute_advanced* is ``True``):
        ``isolation_distance`` — per-cluster dict.
        ``l_ratio`` — per-cluster dict.
        ``d_prime`` — per-cluster dict.
        ``peak_amplitude_snr`` — per-cluster dict.
        ``waveform_stability`` — per-cluster dict.
        ``amplitude_drift`` — per-cluster dict.
        ``fraction_missing`` — per-cluster dict.

    Args:
        data: ``SortingData`` container.
        cluster_labels: Cluster assignment per spike.
        refractory_period: Refractory period in seconds.
        compute_advanced: Include isolation distance, L-ratio,
            d-prime, waveform stability, amplitude drift,
            peak amplitude SNR, and fraction-missing metrics.

    Returns:
        Dict of metric names to values.
    """
    if np.unique(cluster_labels).size < 2:
        raise ValueError("At least 2 clusters are required for evaluation.")
    wv = data.waveforms
    st = data.spike_times

    sil_mean = float(silhouette_score(wv, cluster_labels))
    neg_sil = neg_silhouette_score(wv, cluster_labels, relative=True)
    abs_rpvs = rpvs(st, cluster_labels, refractory_period=refractory_period,
                    relative=False, all_clusters=True)
    rel_rpvs = rpvs(st, cluster_labels, refractory_period=refractory_period,
                    relative=True, all_clusters=True)
    snr_w = calc_weighted_snr(wv, cluster_labels)

    snr_per = {}
    for cl in np.unique(cluster_labels):
        snr_per[int(cl)] = est_snr(wv[cluster_labels == cl])

    result = {
        "neg_silhouette_rel": neg_sil,
        "silhouette_mean": sil_mean,
        "abs_rpvs": abs_rpvs,
        "rel_rpvs": rel_rpvs,
        "snr_weighted": snr_w,
        "snr_per_cluster": snr_per,
    }

    if compute_advanced:
        cl_kw = {"cluster_labels": cluster_labels}
        _adv: list[tuple[str, object, tuple, dict]] = [
            ("isolation_distance", isolation_distance, (wv, cluster_labels), {}),
            ("l_ratio", l_ratio, (wv, cluster_labels), {}),
            ("d_prime", d_prime, (wv, cluster_labels), {}),
            ("peak_amplitude_snr", peak_amplitude_snr, (wv,), cl_kw),
            ("waveform_stability", waveform_stability, (st, wv), cl_kw),
            ("amplitude_drift", amplitude_drift, (wv,), cl_kw),
            ("fraction_missing", fraction_missing, (wv,), cl_kw),
        ]
        for key, func, args, kwargs in _adv:
            try:
                result[key] = func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Advanced metric '{key}' failed: {exc}",
                    stacklevel=2,
                )
                result[key] = np.nan

    return result


def evaluate_os_per_cluster(
    data: SortingData,
    cluster_labels: npt.NDArray[np.int64],
    bin_size: float = 0.01,
    rng: np.random.Generator | int | None = None,
) -> dict[int, dict] | None:
    """Orientation-selectivity metrics for every cluster.

    Requires the ``tuning`` package to be importable.

    Args:
        data: ``SortingData`` container (must have angles).
        cluster_labels: Cluster labels per spike.
        bin_size: PSTH bin width (seconds).
        rng: Generator, int seed, or ``None`` forwarded to
            :func:`get_os_metrics` so any per-cluster bootstrap CIs are
            reproducible.  When a ``Generator`` is passed it is shared
            across clusters, so each successive cluster advances the
            same stream.

    Returns:
        ``{cluster_id: {metric_name: value, ...}, ...}`` or ``None``
        if tuning is not available.
    """
    result: dict[int, dict] = {}
    for cl in np.unique(cluster_labels):
        result[int(cl)] = get_os_metrics(
            spike_times=data.spike_times,
            trials=data.trials,
            angles=data.angles,
            cluster_labels=cluster_labels,
            cluster_id=int(cl),
            all_clusters=False,
            bin_size=bin_size,
            stim_window=data.stim_window,
            stim_frequency=data.stim_frequency,
            return_verbose=1,
            rng=rng,
        )
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_sorting_pipeline(
    data: SortingData,
    n_clusters: int | None = None,
    k_range: Sequence[int] = range(2, 6),
    rng: np.random.Generator | int | None = None,
    n_init: str | int = "auto",
    refractory_period: float = 0.001,
    bin_size: float = 0.01,
    compute_os: bool = True,
    plot: bool = True,
    invert_waveforms: bool = True,
    preprocess: PreprocessMode = "zscore_pca",
    pca_components: int | float | None = None,
) -> SortingResult:
    """End-to-end spike sorting pipeline.

    Steps:
        1. If *n_clusters* is ``None``, scan *k_range* via silhouette.
        2. Run KMeans with the chosen *k*.
        3. Compute sorting-quality metrics.
        4. (Optional) Compute per-cluster orientation selectivity.
        5. (Optional) Produce a diagnostic summary figure.

    Args:
        data: ``SortingData`` container.
        n_clusters: Fixed cluster count.  ``None`` → auto-select.
        k_range: Candidate ks when auto-selecting.
        rng: Generator, int seed, or ``None`` for KMeans reproducibility.
        n_init: KMeans initialisations.
        refractory_period: Refractory period for RPV computation (s).
        bin_size: PSTH bin width for OS metrics (s).
        compute_os: Whether to compute orientation-selectivity metrics.
        plot: Whether to produce the summary figure.
        invert_waveforms: Negate waveforms in the plot.
        preprocess: Feature transformation applied before clustering.
            The same mode is used for both *find_optimal_k* and
            *sort_spikes*, so silhouette scores correspond to the
            actual clustering space.
        pca_components: Component count or variance ratio for the PCA
            mode (ignored otherwise).

    Returns:
        ``SortingResult`` with all outputs.  Quality metrics that
        operate on raw waveforms (SNR, isolation distance, etc.) are
        always computed on ``data.waveforms`` regardless of the
        preprocessing mode.
    """
    # Coerce rng once so all downstream sklearn calls share the same seed
    # (and so the metadata reflects what was actually used).
    seed = _as_seed(rng)

    # --- 1. Determine k ---
    k_search: dict | None = None
    if n_clusters is None:
        n_clusters, k_search = find_optimal_k(
            data.waveforms, k_range=k_range,
            rng=seed, n_init=n_init,
            preprocess=preprocess, pca_components=pca_components,
        )

    # --- 2. Cluster ---
    cluster_labels, km_model = sort_spikes(
        data.waveforms, n_clusters=n_clusters,
        rng=seed, n_init=n_init,
        preprocess=preprocess, pca_components=pca_components,
    )

    # --- 3. Quality ---
    quality = evaluate_sorting(data, cluster_labels,
                               refractory_period=refractory_period)

    # --- 4. OS metrics ---
    os_metrics: dict[int, dict] | None = None
    if compute_os and data.angles is not None and len(data.angles) > 0:
        os_metrics = evaluate_os_per_cluster(
            data, cluster_labels, bin_size=bin_size,
        )

    # --- 5. Plot ---
    if plot:
        plot_sorting_summary(
            data, cluster_labels, invert_waveforms=invert_waveforms,
        )
        if k_search is not None:
            plot_k_search(k_search)
        plt.show()

    return SortingResult(
        cluster_labels=cluster_labels,
        n_clusters=n_clusters,
        quality=quality,
        os_metrics=os_metrics,
        k_search=k_search,
        metadata={
            "rng": seed,
            "n_init": n_init,
            "kmeans_inertia": float(km_model.inertia_),
            "preprocess": preprocess,
            "pca_components": pca_components,
        },
    )
