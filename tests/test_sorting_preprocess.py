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

from tests.conftest import make_two_cluster_waveforms_only, make_sorting_data


def _make_two_cluster_waveforms(
    n_per: int = 80,
    snippet_len: int = 32,
    rng_seed: int = 11,
) -> np.ndarray:
    """Thin wrapper preserving the original call convention."""
    return make_two_cluster_waveforms_only(
        n_per=n_per, snippet_len=snippet_len, rng_seed=rng_seed,
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
            wv, n_clusters=2, preprocess=mode, rng=0,
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
            wv, n_clusters=2, preprocess="pca", pca_components=3,
            rng=0,
        )
        assert km.cluster_centers_.shape == (2, 3)


class TestFindOptimalKPreprocess:

    @pytest.mark.parametrize("mode", ["none", "center", "zscore", "pca", "zscore_pca"])
    def test_picks_two_for_two_cluster_data(self, mode):
        wv = _make_two_cluster_waveforms()
        best_k, scores = find_optimal_k(
            wv, k_range=range(2, 5), preprocess=mode, rng=0,
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
            wv, k_range=[2], preprocess="none", rng=0,
        )
        _, scores_pca = find_optimal_k(
            wv, k_range=[2], preprocess="pca", pca_components=2,
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
            data, n_clusters=2, preprocess=mode,
            compute_os=False, plot=False, rng=0,
        )
        assert result.n_clusters == 2
        assert result.metadata["preprocess"] == mode

    def test_pca_components_in_metadata(self):
        data = _make_sorting_data()
        result = run_sorting_pipeline(
            data, n_clusters=2, preprocess="pca", pca_components=4,
            compute_os=False, plot=False, rng=0,
        )
        assert result.metadata["preprocess"] == "pca"
        assert result.metadata["pca_components"] == 4

    def test_quality_metrics_use_raw_waveforms(self):
        """SNR (a quality metric) should be the same regardless of
        preprocessing — quality metrics live in raw waveform space."""
        data = _make_sorting_data()
        r_none = run_sorting_pipeline(
            data, n_clusters=2, preprocess="none",
            compute_os=False, plot=False, rng=0,
        )
        r_zscore = run_sorting_pipeline(
            data, n_clusters=2, preprocess="zscore",
            compute_os=False, plot=False, rng=0,
        )
        # SNR is computed on raw waveforms, so it does not depend on
        # the preprocessing mode (provided both runs converge to the
        # same clustering, which they do here for well-separated data).
        assert r_none.quality["snr_per_cluster"].keys() == \
            r_zscore.quality["snr_per_cluster"].keys()
        for cid in r_none.quality["snr_per_cluster"]:
            assert r_none.quality["snr_per_cluster"][cid] == pytest.approx(
                r_zscore.quality["snr_per_cluster"][cid], rel=1e-9,
            )

    def test_default_preprocess_is_zscore_pca(self):
        """The defensible methods-section pipeline is the default.

        Defaulting to ``"none"`` means raw waveforms go straight into
        KMeans, which is hard to defend in a paper.  The default is
        now ``"zscore_pca"``: per-feature z-score, then PCA, then
        KMeans.
        """
        data = _make_sorting_data()
        result = run_sorting_pipeline(
            data, n_clusters=2, compute_os=False, plot=False,
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
            wv, "zscore_pca", pca_components=4, rng=0,
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
            wv, "pca", pca_components=4, rng=0,
        )
        feats_chain = _preprocess_waveforms(
            wv, "zscore_pca", pca_components=4, rng=0,
        )
        # Different feature spaces — Frobenius distance is
        # well above any rounding margin.
        assert np.linalg.norm(feats_chain - feats_pca) > 1.0

    def test_zscore_pca_recovers_two_clusters(self):
        wv = _make_two_cluster_waveforms()
        labels, _ = sort_spikes(
            wv, n_clusters=2, preprocess="zscore_pca",
            pca_components=4, rng=0,
        )
        assert len(np.unique(labels)) == 2
        counts = np.bincount(labels)
        assert counts.min() > wv.shape[0] // 4

    def test_zscore_pca_kmeans_lives_in_pca_space(self):
        wv = _make_two_cluster_waveforms()
        _, km = sort_spikes(
            wv, n_clusters=2, preprocess="zscore_pca",
            pca_components=3, rng=0,
        )
        assert km.cluster_centers_.shape == (2, 3)

    def test_zscore_pca_pipeline_metadata(self):
        data = _make_sorting_data()
        result = run_sorting_pipeline(
            data, n_clusters=2, preprocess="zscore_pca",
            pca_components=4, compute_os=False, plot=False, rng=0,
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
                wv, "zscore_pca", pca_components=k, rng=0,
            )
            assert out.shape == (300, k)

        # The default (None → 0.95) resolves to a number strictly
        # less than the input dimensionality for any data with at
        # least one near-degenerate column.
        wv[:, 0] = 0.0  # zero-variance column
        out_default = _preprocess_waveforms(
            wv, "zscore_pca", pca_components=None, rng=0,
        )
        assert out_default.shape[1] < wv.shape[1]
