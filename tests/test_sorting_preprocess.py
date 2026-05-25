"""Tests for the waveform preprocessing modes in sorting.sorting.

Run with:
    python -m pytest tests/test_sorting_preprocess.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.sorting.io_util import SortingData
from neural_cca.sorting.sorting import (
    _preprocess_waveforms,
    find_optimal_k,
    run_sorting_pipeline,
    sort_spikes,
)
from tests.conftest import make_sorting_data, make_two_cluster_waveforms_only


def _make_two_cluster_waveforms(
    n_per: int = 80,
    snippet_len: int = 32,
    rng_seed: int = 11,
) -> np.ndarray:
    """Thin wrapper preserving the original call convention."""
    return make_two_cluster_waveforms_only(
        n_per=n_per,
        snippet_len=snippet_len,
        rng_seed=rng_seed,
    )


def _make_sorting_data(n_per: int = 80, snippet_len: int = 32) -> SortingData:
    return make_sorting_data(n_per=n_per, snippet_len=snippet_len)


# ======================================================================
# _preprocess_waveforms
# ======================================================================


class TestPreprocessWaveforms:
    def test_none_returns_original(self):
        wv = _make_two_cluster_waveforms()
        out = _preprocess_waveforms(wv, "none")
        assert out is wv

    def test_center_zero_mean(self):
        wv = _make_two_cluster_waveforms()
        out = _preprocess_waveforms(wv, "center")
        assert out.shape == wv.shape
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)

    def test_zscore_unit_variance(self):
        wv = _make_two_cluster_waveforms()
        out = _preprocess_waveforms(wv, "zscore")
        assert out.shape == wv.shape
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-12)

    def test_zscore_handles_zero_variance_column(self):
        """A constant column must not produce NaN/Inf."""
        wv = _make_two_cluster_waveforms().copy()
        wv[:, 0] = 7.0  # constant column
        out = _preprocess_waveforms(wv, "zscore")
        assert np.all(np.isfinite(out))
        # The constant column gets centred to 0 and divided by 1.
        np.testing.assert_allclose(out[:, 0], 0.0, atol=1e-12)

    def test_pca_default_reduces_dimensionality(self):
        wv = _make_two_cluster_waveforms()
        out = _preprocess_waveforms(wv, "pca")
        # 0.95 variance ratio should reduce dim below the snippet length
        assert out.shape[0] == wv.shape[0]
        assert out.shape[1] < wv.shape[1]

    def test_pca_explicit_components(self):
        wv = _make_two_cluster_waveforms()
        out = _preprocess_waveforms(wv, "pca", pca_components=3)
        assert out.shape == (wv.shape[0], 3)

    def test_unknown_mode_raises(self):
        wv = _make_two_cluster_waveforms()
        with pytest.raises(ValueError, match="Unknown preprocess mode"):
            _preprocess_waveforms(wv, "wavelet")  # type: ignore[arg-type]


# ======================================================================
# find_optimal_k & sort_spikes with preprocess
# ======================================================================


class TestSortSpikesPreprocess:
    @pytest.mark.parametrize("mode", ["none", "center", "zscore", "pca", "zscore_pca"])
    def test_recovers_two_clusters(self, mode):
        wv = _make_two_cluster_waveforms()
        labels, km = sort_spikes(
            wv,
            n_clusters=2,
            preprocess=mode,
            rng=0,
        )
        assert labels.shape == (wv.shape[0],)
        # Both clusters should be discovered.
        assert len(np.unique(labels)) == 2
        # Each cluster should be roughly the right size — i.e. the
        # split should respect the underlying templates, not produce
        # one giant cluster and one tiny one.
        counts = np.bincount(labels)
        assert counts.min() > wv.shape[0] // 4

    def test_pca_kmeans_lives_in_pca_space(self):
        """KMeans centres should match the PCA-reduced feature dim."""
        wv = _make_two_cluster_waveforms()
        labels, km = sort_spikes(
            wv,
            n_clusters=2,
            preprocess="pca",
            pca_components=3,
            rng=0,
        )
        assert km.cluster_centers_.shape == (2, 3)


class TestFindOptimalKPreprocess:
    @pytest.mark.parametrize("mode", ["none", "center", "zscore", "pca", "zscore_pca"])
    def test_picks_two_for_two_cluster_data(self, mode):
        wv = _make_two_cluster_waveforms()
        best_k, scores = find_optimal_k(
            wv,
            k_range=range(2, 5),
            preprocess=mode,
            rng=0,
        )
        assert best_k == 2
        assert set(scores.keys()) == {2, 3, 4}

    def test_silhouette_computed_in_feature_space(self):
        """Silhouette scores must come from the preprocessed space.

        With PCA(2 components) the silhouette is evaluated in 2-D, which
        is a different number than the silhouette of the same labels in
        the raw 32-D space.
        """
        wv = _make_two_cluster_waveforms()
        _, scores_raw = find_optimal_k(
            wv,
            k_range=[2],
            preprocess="none",
            rng=0,
        )
        _, scores_pca = find_optimal_k(
            wv,
            k_range=[2],
            preprocess="pca",
            pca_components=2,
            rng=0,
        )
        # The scores should differ — they're computed in different
        # spaces.  (Both should still be positive for well-separated
        # clusters.)
        assert scores_raw[2] != scores_pca[2]
        assert scores_pca[2] > 0


# ======================================================================
# run_sorting_pipeline preprocess threading
# ======================================================================


class TestPipelinePreprocess:
    @pytest.mark.parametrize("mode", ["none", "center", "zscore", "pca", "zscore_pca"])
    def test_pipeline_runs(self, mode):
        data = _make_sorting_data()
        result = run_sorting_pipeline(
            data,
            n_clusters=2,
            preprocess=mode,
            compute_os=False,
            plot=False,
            rng=0,
        )
        assert result.n_clusters == 2
        assert result.metadata["preprocess"] == mode

    def test_pca_components_in_metadata(self):
        data = _make_sorting_data()
        result = run_sorting_pipeline(
            data,
            n_clusters=2,
            preprocess="pca",
            pca_components=4,
            compute_os=False,
            plot=False,
            rng=0,
        )
        assert result.metadata["preprocess"] == "pca"
        assert result.metadata["pca_components"] == 4

    def test_amplitude_metrics_use_raw_waveforms(self):
        """Amplitude-based metrics (SNR, peak-amplitude SNR, drift, …)
        live in raw-waveform space regardless of *preprocess*.

        Voltage amplitude is only meaningful in the raw signal, so
        these metrics ignore the preprocessing pipeline.  Feature-
        space metrics (silhouette, isolation distance, L-ratio,
        d-prime) by contrast use the same space the clustering ran in
        — see :func:`test_feature_space_metrics_depend_on_preprocess`.
        """
        data = _make_sorting_data()
        r_none = run_sorting_pipeline(
            data,
            n_clusters=2,
            preprocess="none",
            compute_os=False,
            plot=False,
            rng=0,
        )
        r_zscore = run_sorting_pipeline(
            data,
            n_clusters=2,
            preprocess="zscore",
            compute_os=False,
            plot=False,
            rng=0,
        )
        assert (
            r_none.quality["snr_per_cluster"].keys() == r_zscore.quality["snr_per_cluster"].keys()
        )
        for cid in r_none.quality["snr_per_cluster"]:
            assert r_none.quality["snr_per_cluster"][cid] == pytest.approx(
                r_zscore.quality["snr_per_cluster"][cid],
                rel=1e-9,
            )

    def test_feature_space_metrics_depend_on_preprocess(self):
        """Silhouette is now reported on the *preprocessed* space so it
        matches the value k-selection sees.  Two different
        preprocessing modes on the same well-separated data therefore
        give two different ``silhouette_mean`` values.
        """
        data = _make_sorting_data()
        r_none = run_sorting_pipeline(
            data,
            n_clusters=2,
            preprocess="none",
            compute_os=False,
            plot=False,
            rng=0,
        )
        r_pca = run_sorting_pipeline(
            data,
            n_clusters=2,
            preprocess="pca",
            pca_components=2,
            compute_os=False,
            plot=False,
            rng=0,
        )
        # Same labels (well-separated data) but the silhouette space
        # differs, so the scalar must differ too.
        assert r_none.quality["silhouette_mean"] != r_pca.quality["silhouette_mean"]

    def test_pipeline_silhouette_matches_k_search(self):
        """When the pipeline auto-selects k, the silhouette reported in
        ``quality`` must equal the score recorded in ``k_search`` for
        the chosen k.  Before the fix the two numbers lived in
        different feature spaces and silently disagreed.
        """
        data = _make_sorting_data()
        result = run_sorting_pipeline(
            data,
            k_range=range(2, 4),
            preprocess="zscore_pca",
            pca_components=3,
            compute_os=False,
            plot=False,
            rng=0,
        )
        chosen_k = result.n_clusters
        assert result.k_search is not None
        assert result.quality["silhouette_mean"] == pytest.approx(
            result.k_search[chosen_k], abs=1e-10
        )

    def test_pipeline_forwards_rng_to_os_bootstrap(self):
        """``rng`` must propagate into per-cluster bootstrap CIs.

        Before the fix, ``run_sorting_pipeline`` coerced ``rng`` for
        sklearn but never handed it to ``evaluate_os_per_cluster``, so
        any bootstrap CI computed downstream was unseeded.  We can't
        easily request a CI through the pipeline directly, but we can
        verify the seed reaches ``evaluate_os_per_cluster`` by patching
        it and reading what was passed.
        """
        from unittest.mock import patch

        from neural_cca.sorting import sorting as _sorting_mod

        data = _make_sorting_data()
        # Wrap the real implementation so the pipeline still produces a
        # SortingResult; record the ``rng`` kwarg actually passed.
        with patch.object(
            _sorting_mod,
            "evaluate_os_per_cluster",
            wraps=_sorting_mod.evaluate_os_per_cluster,
        ) as mock_eval:
            run_sorting_pipeline(
                data,
                n_clusters=2,
                compute_os=True,
                plot=False,
                rng=12345,
            )
        assert mock_eval.call_count == 1
        kwargs = mock_eval.call_args.kwargs
        assert "rng" in kwargs, (
            "run_sorting_pipeline must pass rng to evaluate_os_per_cluster "
            "for per-cluster bootstrap reproducibility."
        )
        assert kwargs["rng"] == 12345

    def test_default_preprocess_is_zscore_pca(self):
        """The defensible methods-section pipeline is the default.

        Defaulting to ``"none"`` means raw waveforms go straight into
        KMeans, which is hard to defend in a paper.  The default is
        now ``"zscore_pca"``: per-feature z-score, then PCA, then
        KMeans.
        """
        data = _make_sorting_data()
        result = run_sorting_pipeline(
            data,
            n_clusters=2,
            compute_os=False,
            plot=False,
            rng=0,
        )
        assert result.metadata["preprocess"] == "zscore_pca"


# ======================================================================
# Chained zscore_pca mode
# ======================================================================


class TestZscorePcaChain:
    """Tests for the chained zscore → PCA → KMeans pipeline."""

    def test_zscore_pca_zscores_first(self):
        """A constant per-feature offset must not survive z-scoring.

        The whole point of the chained mode is that PCA sees
        unit-variance, zero-mean features.  We confirm by adding a
        large per-feature offset and checking the resulting PCA
        scores are still zero-mean (so the offset has been removed).
        """
        rng = np.random.default_rng(7)
        wv = rng.standard_normal((200, 32)) + np.linspace(0, 50, 32)
        out = _preprocess_waveforms(
            wv,
            "zscore_pca",
            pca_components=4,
            rng=0,
        )
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)

    def test_zscore_pca_differs_from_plain_pca(self):
        """Chained pipeline must differ from PCA-only on noisy data.

        On data where one feature has much larger variance than the
        others (the typical waveform-snippet pathology — peak sample
        dominates), plain PCA picks the high-variance axis as PC1
        while ``zscore_pca`` first equalises the variances, producing
        a meaningfully different feature space.
        """
        rng = np.random.default_rng(11)
        wv = rng.standard_normal((300, 16))
        wv[:, 7] *= 50.0  # one dominant axis
        feats_pca = _preprocess_waveforms(
            wv,
            "pca",
            pca_components=4,
            rng=0,
        )
        feats_chain = _preprocess_waveforms(
            wv,
            "zscore_pca",
            pca_components=4,
            rng=0,
        )
        # Different feature spaces — Frobenius distance is
        # well above any rounding margin.
        assert np.linalg.norm(feats_chain - feats_pca) > 1.0

    def test_zscore_pca_recovers_two_clusters(self):
        wv = _make_two_cluster_waveforms()
        labels, _ = sort_spikes(
            wv,
            n_clusters=2,
            preprocess="zscore_pca",
            pca_components=4,
            rng=0,
        )
        assert len(np.unique(labels)) == 2
        counts = np.bincount(labels)
        assert counts.min() > wv.shape[0] // 4

    def test_zscore_pca_kmeans_lives_in_pca_space(self):
        wv = _make_two_cluster_waveforms()
        _, km = sort_spikes(
            wv,
            n_clusters=2,
            preprocess="zscore_pca",
            pca_components=3,
            rng=0,
        )
        assert km.cluster_centers_.shape == (2, 3)

    def test_zscore_pca_pipeline_metadata(self):
        data = _make_sorting_data()
        result = run_sorting_pipeline(
            data,
            n_clusters=2,
            preprocess="zscore_pca",
            pca_components=4,
            compute_os=False,
            plot=False,
            rng=0,
        )
        assert result.metadata["preprocess"] == "zscore_pca"
        assert result.metadata["pca_components"] == 4

    def test_pca_components_is_explicit_parameter(self):
        """``pca_components`` controls the output dimensionality.

        The contract is that dimensionality is an explicit parameter,
        not a magic constant — passing different integer values must
        produce feature spaces of the corresponding dimension, and
        the default ``None`` resolves to the documented 0.95 variance
        ratio (which is the value the writer reads when serialising).
        """
        rng = np.random.default_rng(0)
        wv = rng.standard_normal((300, 32))

        # Explicit integer component counts produce exactly that many
        # PCs in the output.
        for k in (2, 5, 8):
            out = _preprocess_waveforms(
                wv,
                "zscore_pca",
                pca_components=k,
                rng=0,
            )
            assert out.shape == (300, k)

        # The default (None → 0.95) resolves to a number strictly
        # less than the input dimensionality for any data with at
        # least one near-degenerate column.
        wv[:, 0] = 0.0  # zero-variance column
        out_default = _preprocess_waveforms(
            wv,
            "zscore_pca",
            pca_components=None,
            rng=0,
        )
        assert out_default.shape[1] < wv.shape[1]


# ======================================================================
# Single-cluster (k=1) sorting path
# ======================================================================


class TestSingleCluster:
    """Pin the ``n_clusters=1`` behaviour added to support
    pre-isolated single-unit channels (Kilosort export, manual
    curation) and the ``min_silhouette`` soft fallback in
    auto-select."""

    def test_pipeline_runs_with_n_clusters_1(self):
        """``n_clusters=1`` runs end-to-end and NaN-fills the
        feature-space-separation metrics with a single
        RuntimeWarning."""
        data = make_sorting_data()
        with pytest.warns(RuntimeWarning, match="k=1"):
            r = run_sorting_pipeline(
                data,
                n_clusters=1,
                compute_os=False,
                plot=False,
                rng=0,
            )
        assert r.n_clusters == 1
        assert np.unique(r.cluster_labels).tolist() == [0]
        # Silhouette family → NaN by construction at k=1.
        assert np.isnan(r.quality["silhouette_mean"])
        assert np.isnan(r.quality["neg_silhouette_rel"])
        # Isolation / L-ratio / d-prime existed before this change
        # with per-cluster NaN guards that already triggered at k=1;
        # verify they still NaN out.
        for key in ("isolation_distance", "l_ratio", "d_prime"):
            assert all(np.isnan(v) for v in r.quality[key].values())
        # Amplitude-based metrics + RPVs stay numeric.
        snrs = r.quality["snr_per_cluster"]
        assert len(snrs) == 1
        assert not np.isnan(next(iter(snrs.values())))
        assert isinstance(r.quality["abs_rpvs"], int)
        assert isinstance(r.quality["snr_weighted"], float)

    def test_find_optimal_k_rejects_k_less_than_2(self):
        """``find_optimal_k`` refuses k<2 with a clear pointer at the
        single-cluster paths.  Before the change, the function
        crashed inside sklearn's ``silhouette_score`` with a less
        helpful message."""
        from neural_cca.sorting.sorting import find_optimal_k

        wv = make_two_cluster_waveforms_only()
        with pytest.raises(ValueError, match="k >= 2"):
            find_optimal_k(wv, k_range=range(1, 5), rng=0)
        with pytest.raises(ValueError, match="k >= 2"):
            find_optimal_k(wv, k_range=[0, 2, 3], rng=0)

    def test_pipeline_auto_select_with_k1_in_range_still_raises(self):
        """The pipeline-internal silhouette search inherits the same
        guard, so passing a k_range that starts at 1 produces the
        same ValueError instead of a sklearn crash."""
        data = make_sorting_data()
        with pytest.raises(ValueError, match="k >= 2"):
            run_sorting_pipeline(
                data,
                k_range=range(1, 4),
                compute_os=False,
                plot=False,
                rng=0,
            )

    def test_k1_with_os_metrics(self):
        """OS metrics still compute at k=1 — the per-cluster loop
        yields exactly one entry, keyed by the single label 0,
        with all returned values in their documented ranges."""
        data = make_sorting_data()
        with pytest.warns(RuntimeWarning, match="k=1"):
            r = run_sorting_pipeline(
                data,
                n_clusters=1,
                compute_os=True,
                plot=False,
                rng=0,
            )
        assert r.os_metrics is not None
        assert list(r.os_metrics.keys()) == [0]
        m = r.os_metrics[0]
        # The standard set of OS keys is present.
        for key in ("osi", "dsi", "preferred_orientation"):
            assert key in m
        # OSI / DSI live in [0, 1] when finite; preferred angle on
        # the orientation circle [0, 180).  Either of these can be
        # NaN for a silent / untuned unit, but never out of range.
        for key in ("osi", "dsi"):
            v = m[key]
            assert np.isnan(v) or 0.0 <= v <= 1.0, f"{key}={v} out of [0, 1]"
        pref = m["preferred_orientation"]
        assert np.isnan(pref) or 0.0 <= pref < 180.0, f"preferred={pref} out of [0, 180)"

    def test_k1_zarr_roundtrip(self, tmp_path):
        """Single-cluster sorting round-trips through both zarr layouts
        with bit-identical labels, spike times, waveforms, and angles."""
        from neural_cca.sorting.io_util import (
            read_zarr_sorting,
            to_zarr_clustered,
            to_zarr_flat,
        )

        zarr = pytest.importorskip("zarr")  # noqa: F841 (skip if not installed)

        data = make_sorting_data()
        with pytest.warns(RuntimeWarning, match="k=1"):
            r = run_sorting_pipeline(
                data,
                n_clusters=1,
                compute_os=False,
                plot=False,
                rng=0,
            )
        for layout, fn in [("flat", to_zarr_flat), ("clustered", to_zarr_clustered)]:
            store = tmp_path / f"k1_{layout}.zarr"
            fn(r, data, store)
            r2, d2 = read_zarr_sorting(store)
            assert r2.n_clusters == 1
            np.testing.assert_array_equal(r.cluster_labels, r2.cluster_labels)
            # Raw input data also survives the round-trip.
            np.testing.assert_allclose(d2.waveforms, data.waveforms)
            np.testing.assert_allclose(d2.spike_times, data.spike_times)
            np.testing.assert_array_equal(d2.trials, data.trials)
            np.testing.assert_allclose(d2.angles, data.angles)
            # Quality NaNs survive round-trip cleanly.  JSON-NaN
            # normalises to ``None`` in the attrs path on modern zarr,
            # so the ``is None`` check must come first to short-circuit
            # before ``np.isnan`` (which would TypeError on ``None``).
            sil = r2.quality["silhouette_mean"]
            assert sil is None or np.isnan(sil)
            # min_silhouette metadata round-trips
            assert "min_silhouette" in r2.metadata
            assert "min_silhouette_triggered" in r2.metadata

    def test_min_silhouette_fallback_to_k1(self):
        """When no k>=2 candidate clears ``min_silhouette``, the
        pipeline declines to split and returns a single cluster.

        Uses a uniformly-Gaussian feature distribution where there
        is no genuine cluster structure, so every silhouette score
        comes out near zero — below any reasonable threshold.
        """
        rng_np = np.random.default_rng(0)
        n = 200
        wv = rng_np.standard_normal((n, 32)).astype(np.float64)
        data = SortingData(
            waveforms=wv,
            spike_times=rng_np.uniform(0.5, 2.5, n),
            trials=rng_np.integers(0, 12, n).astype(np.int64),
            angles=np.linspace(0, 330, 12),
            n_trials=12,
            stim_window=(0.5, 2.5),
        )
        with pytest.warns(RuntimeWarning, match="k=1"):
            r = run_sorting_pipeline(
                data,
                k_range=range(2, 5),
                min_silhouette=0.5,  # unreachable on isotropic noise
                compute_os=False,
                plot=False,
                rng=0,
            )
        assert r.n_clusters == 1
        assert r.metadata["min_silhouette"] == 0.5
        assert r.metadata["min_silhouette_triggered"] is True
        # k_search is still populated so the user can audit the search.
        assert r.k_search is not None
        assert set(r.k_search.keys()) == {2, 3, 4}

    def test_min_silhouette_does_not_trigger_on_real_clusters(self):
        """On well-separated data the silhouette clears even
        moderate thresholds, so the fallback does NOT fire."""
        data = make_sorting_data()
        result = run_sorting_pipeline(
            data,
            k_range=range(2, 4),
            min_silhouette=0.05,  # low bar, well-separated data clears it
            compute_os=False,
            plot=False,
            rng=0,
        )
        assert result.n_clusters >= 2
        assert result.metadata["min_silhouette_triggered"] is False

    def test_min_silhouette_ignored_when_n_clusters_explicit(self):
        """``min_silhouette`` is an *auto-select* policy — it is
        ignored when the user fixes ``n_clusters`` explicitly."""
        data = make_sorting_data()
        # Even with an unreachable threshold, n_clusters=2 wins.
        result = run_sorting_pipeline(
            data,
            n_clusters=2,
            min_silhouette=0.99,
            compute_os=False,
            plot=False,
            rng=0,
        )
        assert result.n_clusters == 2
        assert result.metadata["min_silhouette_triggered"] is False


# ======================================================================
# SortingData construction-time validation (regression: stim_window
# with zero/negative duration used to silently divide-by-zero in
# downstream firing-rate calculations).
# ======================================================================


class TestSortingDataValidation:
    def _minimal_arrays(self):
        return dict(
            waveforms=np.zeros((5, 8), dtype=np.float64),
            spike_times=np.zeros(5, dtype=np.float64),
            trials=np.zeros(5, dtype=np.int64),
            angles=np.array([0.0], dtype=np.float64),
        )

    def test_zero_duration_stim_window_raises(self):
        with pytest.raises(ValueError, match="onset < end"):
            SortingData(**self._minimal_arrays(), stim_window=(2.5, 2.5))

    def test_inverted_stim_window_raises(self):
        with pytest.raises(ValueError, match="onset < end"):
            SortingData(**self._minimal_arrays(), stim_window=(3.0, 0.5))

    def test_valid_stim_window_accepted(self):
        """Sanity: the validation doesn't break the happy path."""
        data = SortingData(**self._minimal_arrays(), stim_window=(0.5, 2.5))
        assert data.stim_window == (0.5, 2.5)
        assert data.stimulus_duration == 2.0
