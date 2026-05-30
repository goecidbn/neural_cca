"""Spike sorting pipeline — clustering, evaluation, orchestration.

End-to-end workflow: cluster waveforms, select optimal *k*, evaluate
sorting quality, compute orientation-selectivity metrics per cluster,
and produce a diagnostic summary figure.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Cross-package import for orientation-selectivity metrics.  This is
# only used by ``evaluate_os_per_cluster``; if ``tuning`` is
# unavailable for any reason at install time the import will fail at
# module load and the user gets a clean error.  ``tuning`` is
# part of the same distribution so this is always installed.
from .._utils import make_rng
from ..tuning.tuning import get_os_metrics
from .containers import SortingData, SortingResult
from .metrics import (
    amplitude_drift,
    calc_weighted_snr,
    contamination_rate_hill,
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
from .plotting import plot_k_search, plot_sorting_summary

__all__ = [
    "SortingResult",
    "find_optimal_k",
    "sort_spikes",
    "evaluate_sorting",
    "evaluate_os_per_cluster",
    "run_sorting_pipeline",
    "PreprocessMode",
]


PreprocessMode = Literal["none", "center", "zscore", "pca", "zscore_pca"]


def _as_seed(rng: np.random.Generator | int | None) -> int | None:
    """Coerce an rng spec to a **uint32** seed for sklearn estimators.

    sklearn's ``random_state`` must fit in a uint32 (``[0, 2**32)``); a
    :class:`numpy.random.Generator` is not accepted. This helper always
    returns a valid, well-mixed seed:

    * ``int`` (any size, including the bridge's ~128-bit
      ``SeedSequence().entropy`` master seed) → a uint32 **derived via
      ``SeedSequence``**, never the raw integer. A raw 128-bit seed
      raises ``InvalidParameterError`` inside ``KMeans`` / ``PCA``;
      routing through ``SeedSequence`` mixes the entropy down to a
      uint32 *deterministically* (same master seed → same sklearn
      seed), so the recorded provenance seed stays replayable.
    * ``Generator`` → one uint32 drawn from its stream (repeated calls
      with the same Generator therefore yield different sklearn seeds,
      matching the consumed-stream expectation).
    * ``None`` → ``None`` (sklearn falls back to its global RNG).

    See ``CROSS_CHECKS.md`` → *RNG policy*. The
    ``tests/test_rng_policy.py`` round-trip guards this against
    regression.
    """
    if rng is None:
        return None
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(0, 2**31 - 1))
    if isinstance(rng, (int, np.integer)):
        return int(np.random.SeedSequence(int(rng)).generate_state(1, dtype=np.uint32)[0])
    raise TypeError(f"rng must be a Generator, int, or None; got {type(rng).__name__}")


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

    Raises:
        ValueError: When *k_range* contains any value below 2.
            Silhouette is undefined for a single cluster; to force
            ``k=1`` use :func:`sort_spikes` directly or pass
            ``n_clusters=1`` to :func:`run_sorting_pipeline`.  See
            also the ``min_silhouette`` parameter on
            :func:`run_sorting_pipeline` for an auto-select policy
            that *can* fall back to k=1.
    """
    ks = list(k_range)
    _reject_k_below_2(ks)
    seed = _as_seed(rng)
    feats = _preprocess_waveforms(
        waveforms,
        preprocess,
        pca_components=pca_components,
        rng=seed,
    )
    scores: dict[int, float] = {}
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(feats)
        scores[k] = float(silhouette_score(feats, labels))

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best_k, scores


def _reject_k_below_2(ks: Sequence[int]) -> None:
    """Refuse silhouette-based k-search for any ``k < 2``.

    Silhouette score is mathematically undefined for a single
    cluster (sklearn raises ``ValueError`` inside
    ``silhouette_score`` when the unique-label count is 1), so
    rather than crash deep in sklearn we surface a clear message
    pointing at the intended single-cluster path
    (``run_sorting_pipeline(n_clusters=1)`` or
    ``sort_spikes(..., n_clusters=1)``) and at the soft fallback
    (``min_silhouette=`` on ``run_sorting_pipeline``).
    """
    bad = sorted(k for k in ks if k < 2)
    if bad:
        raise ValueError(
            f"find_optimal_k requires every k >= 2 (silhouette is "
            f"undefined for k=1); got k_range containing {bad}. To "
            "force a single cluster, call run_sorting_pipeline(..., "
            "n_clusters=1) or sort_spikes(..., n_clusters=1) "
            "directly.  For auto-selection that can prefer k=1, see "
            "the `min_silhouette` parameter on run_sorting_pipeline."
        )


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
        waveforms,
        preprocess,
        pca_components=pca_components,
        rng=seed,
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
    features: npt.NDArray[np.float64] | None = None,
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
        features: Optional ``(n_spikes, n_features)`` feature matrix
            used for the *feature-space* metrics: silhouette,
            isolation distance, L-ratio, d-prime.  When ``None``
            (default) these metrics fall back to ``data.waveforms``
            for backwards compatibility.  :func:`run_sorting_pipeline`
            forwards the preprocessed feature matrix here so the
            silhouette stored in ``quality`` agrees with the value
            used during k-selection.  Amplitude-based metrics
            (``snr_*``, ``peak_amplitude_snr``,
            ``waveform_stability``, ``amplitude_drift``,
            ``fraction_missing``) always operate on raw waveforms
            because amplitude is only meaningful in voltage space.

    Returns:
        Dict of metric names to values.  Keys are uniform across the
        k=1 and k>=2 paths so downstream schemas (zarr writer, etc.)
        do not have to special-case the cluster count.  At k=1 the
        feature-space-separation metrics — ``silhouette_mean``,
        ``neg_silhouette_rel``, plus the per-cluster
        ``isolation_distance``, ``l_ratio``, and ``d_prime`` entries —
        are filled with ``np.nan`` (they are mathematically
        undefined for a single cluster) and a single
        :class:`RuntimeWarning` is emitted.  Amplitude / shape /
        rate metrics remain well-defined and are computed
        normally.

    Raises:
        ValueError: When ``cluster_labels`` is empty.
    """
    n_uniq = int(np.unique(cluster_labels).size)
    if n_uniq < 1:
        # Truly no labels — this is not "k=1 with a silent unit", it is
        # an empty input, so refuse rather than NaN-fill.
        raise ValueError("cluster_labels is empty.")
    single_cluster = n_uniq == 1

    wv = data.waveforms
    st = data.spike_times
    # Feature-space metrics use ``features`` when provided so they match
    # the space the clustering was actually performed in.  Falling back
    # to ``wv`` keeps the standalone behaviour unchanged.
    feat = wv if features is None else np.asarray(features, dtype=np.float64)

    if single_cluster:
        # Silhouette is undefined for k=1 (sklearn would raise); the
        # per-cluster feature-space metrics route through existing
        # NaN-returning guards (``isolation_distance``, ``l_ratio``,
        # ``d_prime`` already short-circuit when ``len(f_out) == 0``
        # or ``len(unique) < 2``).  We emit one warning per call so
        # the user sees the silhouette-class NaNs are by-design, not
        # a silent failure.
        warnings.warn(
            "Evaluating sorting with k=1: silhouette, "
            "neg_silhouette_rel, isolation_distance, l_ratio, and "
            "d_prime are undefined for a single cluster and will be "
            "filled with NaN.  Amplitude-based metrics "
            "(snr_*, peak_amplitude_snr, waveform_stability, "
            "amplitude_drift, fraction_missing) and the RPV metrics "
            "remain well-defined.",
            RuntimeWarning,
            stacklevel=2,
        )
        sil_mean = float("nan")
        neg_sil = float("nan")
    else:
        sil_mean = float(silhouette_score(feat, cluster_labels))
        neg_sil = neg_silhouette_score(feat, cluster_labels, relative=True)
    abs_rpvs = rpvs(
        st,
        cluster_labels,
        trials=data.trials,
        refractory_period=refractory_period,
        relative=False,
        all_clusters=True,
    )
    rel_rpvs = rpvs(
        st,
        cluster_labels,
        trials=data.trials,
        refractory_period=refractory_period,
        relative=True,
        all_clusters=True,
    )
    snr_w = calc_weighted_snr(wv, cluster_labels)

    snr_per = {}
    for cl in np.unique(cluster_labels):
        snr_per[int(cl)] = est_snr(wv[cluster_labels == cl])

    # Hill 2011 contamination fraction per cluster.  The total
    # recording duration is the trial count times the assumed trial
    # length (``stim_window[1]`` — the trial is taken to span
    # ``[0, end]`` everywhere else in the package).  See the
    # ``contamination_rate_hill`` docstring for the scaling caveats.
    try:
        rec_duration = float(data.n_trials) * float(data.stim_window[1])
        contam = contamination_rate_hill(
            st,
            cluster_labels=cluster_labels,
            trials=data.trials,
            recording_duration=rec_duration,
            refractory_period=refractory_period,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"contamination_rate_hill failed: {exc}; reporting NaN per cluster.",
            stacklevel=2,
        )
        contam = {int(c): float("nan") for c in np.unique(cluster_labels)}

    result = {
        "neg_silhouette_rel": neg_sil,
        "silhouette_mean": sil_mean,
        "abs_rpvs": abs_rpvs,
        "rel_rpvs": rel_rpvs,
        "contamination_rate_hill": contam,
        "snr_weighted": snr_w,
        "snr_per_cluster": snr_per,
    }

    if compute_advanced:
        cl_kw = {"cluster_labels": cluster_labels}
        _adv: list[tuple[str, object, tuple, dict]] = [
            # Feature-space metrics (Mahalanobis / mean separation) —
            # use ``feat`` so they live in the clustering space.
            ("isolation_distance", isolation_distance, (feat, cluster_labels), {}),
            ("l_ratio", l_ratio, (feat, cluster_labels), {}),
            ("d_prime", d_prime, (feat, cluster_labels), {}),
            # Amplitude / shape metrics — always raw waveforms.
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
            reproducible.  A single ``Generator`` is materialised here
            (via :func:`neural_cca._utils.make_rng`) and **shared**
            across every cluster, so the full per-cluster bootstrap
            sequence is reproducible from one integer seed.  An earlier
            version passed the raw argument through, which made integer
            seeds produce *identical* bootstrap streams in every
            cluster (each ``get_os_metrics`` call rebuilt a fresh
            Generator from the same seed).

    Returns:
        ``{cluster_id: {metric_name: value, ...}, ...}`` or ``None``
        if tuning is not available.
    """
    # Materialise once and share across clusters so successive bootstraps
    # advance the same stream.  ``make_rng`` returns the Generator
    # unchanged if one is already passed in, wraps an integer seed in a
    # ``SeedSequence``, and falls back to OS entropy on ``None``.
    boot_rng = make_rng(rng)
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
            rng=boot_rng,
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
    min_silhouette: float | None = None,
) -> SortingResult:
    """End-to-end spike sorting pipeline.

    Steps:
        1. If *n_clusters* is ``None``, scan *k_range* via silhouette.
        2. Run KMeans with the chosen *k*.
        3. Compute sorting-quality metrics.
        4. (Optional) Compute per-cluster orientation selectivity.
        5. (Optional) Produce a diagnostic summary figure.

    Notes
    -----
    **Single-cluster (``k = 1``) mode.** The pipeline accepts
    ``n_clusters=1`` as a first-class supported path.  Use it for:

    * **Pre-isolated channels.** Upstream tooling (Kilosort export,
      manual curation) already restricted a recording to a single
      unit; ``run_sorting_pipeline(data, n_clusters=1)`` then
      computes quality + OS metrics on that lone unit *without*
      re-clustering the waveforms.
    * **Low-density "trust the channel" recordings** where every
      channel is taken to be one cell.
    * **Auto-select fallback.** Setting ``min_silhouette`` triggers
      a fall-back to ``k = 1`` whenever no candidate in *k_range*
      beats the threshold — see the parameter description.

    At ``k = 1`` the silhouette-class quality metrics
    (``silhouette_mean``, ``neg_silhouette_rel``, plus per-cluster
    ``isolation_distance``, ``l_ratio``, ``d_prime``) are filled
    with ``np.nan`` and :func:`evaluate_sorting` emits a single
    :class:`RuntimeWarning` listing exactly which keys are NaN by
    construction.  Everything else (RPV, SNR, peak-amplitude SNR,
    waveform stability, amplitude drift, fraction missing, and the
    full OS-metrics dict if angles are present) is well-defined and
    computed normally.

    Args:
        data: ``SortingData`` container.
        n_clusters: Fixed cluster count.  ``None`` → auto-select via
            silhouette over *k_range*.  ``1`` → skip k-search and
            treat the recording as a single pre-isolated unit (see
            the section above).  ``>= 2`` → cluster into that many
            groups directly.
        k_range: Candidate ks when auto-selecting.  Must start at
            ``>= 2`` — silhouette is undefined for a single cluster,
            so ``k = 1`` is never *searched*; it is only ever
            *chosen* via ``n_clusters=1`` or ``min_silhouette``.
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
        min_silhouette: Soft k-search fallback to ``k = 1``.  When
            ``None`` (default) auto-select always picks the
            best-scoring k from *k_range*.  When set, the chosen k
            from the k-search is overridden with ``1`` if its mean
            silhouette is below *min_silhouette* — i.e. if the data
            does not produce a meaningful clustering, the pipeline
            declines to split it rather than reporting arbitrary
            halves.  The ``k_search`` dict is still recorded so the
            user can see which candidate was tried and what its
            score was.  Only consulted when *n_clusters* is
            ``None``.  Reasonable values lie around ``0.05``–``0.2``;
            anything below ``0`` makes the fallback impossible
            (silhouette is bounded below by −1).

    Returns:
        ``SortingResult`` with all outputs.

        Feature-space metrics (``silhouette_mean``, ``neg_silhouette_rel``,
        ``isolation_distance``, ``l_ratio``, ``d_prime``) are computed
        on the *preprocessed* feature matrix — the same space the
        clustering was performed in — so the ``silhouette_mean`` in
        ``quality`` agrees with the value used by k-selection.
        Amplitude/shape metrics (``snr_*``, ``peak_amplitude_snr``,
        ``waveform_stability``, ``amplitude_drift``,
        ``fraction_missing``) are always computed on raw
        ``data.waveforms`` because amplitude is only meaningful in
        voltage space.  See the section above for the ``k = 1``
        behaviour.

        ``metadata`` records ``min_silhouette`` and the boolean
        ``min_silhouette_triggered`` so a downstream consumer can
        tell whether the chosen k came from the silhouette argmax
        or from the threshold fallback.
    """
    # Coerce rng once so all downstream sklearn calls share the same seed
    # (and so the metadata reflects what was actually used).
    seed = _as_seed(rng)

    # Preprocess once and reuse: clustering and the feature-space
    # quality metrics (silhouette / isolation / L-ratio / d-prime) both
    # read from the same matrix, so the silhouette stored in
    # ``quality`` is the same number that k-selection saw.
    features = _preprocess_waveforms(
        data.waveforms,
        preprocess,
        pca_components=pca_components,
        rng=seed,
    )

    # --- 1. Determine k ---
    k_search: dict | None = None
    min_silhouette_triggered = False
    if n_clusters is None:
        n_clusters, k_search = _find_optimal_k_from_features(
            features,
            k_range=k_range,
            seed=seed,
            n_init=n_init,
        )
        # Soft fallback: if the best silhouette in ``k_range`` did not
        # clear ``min_silhouette``, declare a single cluster.  This is
        # the canonical "data has no separable structure → don't
        # invent any" guard.  We keep ``k_search`` populated so the
        # user can see what was actually tried; the fact that we did
        # *not* pick its argmax is recorded in
        # ``min_silhouette_triggered``.
        if min_silhouette is not None and k_search and k_search[n_clusters] < float(min_silhouette):
            n_clusters = 1
            min_silhouette_triggered = True

    # --- 2. Cluster ---
    # ``KMeans(n_clusters=1)`` is valid sklearn: it returns all-zeros
    # labels and an inertia of the within-data sum of squares to the
    # global mean.  No special-casing required here.
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    cluster_labels = km.fit_predict(features).astype(np.int64)
    km_model = km

    # --- 3. Quality ---
    quality = evaluate_sorting(
        data,
        cluster_labels,
        refractory_period=refractory_period,
        features=features,
    )

    # --- 4. OS metrics ---
    # ``evaluate_os_per_cluster`` iterates ``np.unique(cluster_labels)``,
    # so the k=1 case naturally returns ``{0: <metrics>}``.
    os_metrics: dict[int, dict] | None = None
    if compute_os and data.angles is not None and len(data.angles) > 0:
        os_metrics = evaluate_os_per_cluster(
            data,
            cluster_labels,
            bin_size=bin_size,
            rng=seed,
        )

    # --- 5. Plot ---
    # ``plot_sorting_summary`` already wraps the single-cluster case
    # (``n_cl == 1`` → ``subfigs = [subfigs]``); plotting at k=1 is
    # one row with two/three panels and just works.
    if plot:
        plot_sorting_summary(
            data,
            cluster_labels,
            invert_waveforms=invert_waveforms,
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
            "min_silhouette": (float(min_silhouette) if min_silhouette is not None else None),
            "min_silhouette_triggered": bool(min_silhouette_triggered),
        },
    )


def _find_optimal_k_from_features(
    features: npt.NDArray[np.float64],
    k_range: Sequence[int],
    seed: int | None,
    n_init: str | int,
) -> tuple[int, dict[int, float]]:
    """Silhouette-based k-selection on an already-preprocessed feature matrix.

    Used by :func:`run_sorting_pipeline` so the preprocessing happens
    exactly once.  Public callers should keep using
    :func:`find_optimal_k`, which preprocesses internally.  The same
    ``k >= 2`` guard is enforced here so the pipeline's auto-select
    path produces the same error message as the standalone helper.
    """
    ks = list(k_range)
    _reject_k_below_2(ks)
    scores: dict[int, float] = {}
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
        labels = km.fit_predict(features)
        scores[k] = float(silhouette_score(features, labels))
    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best_k, scores
