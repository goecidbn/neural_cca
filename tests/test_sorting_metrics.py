"""Tests for advanced spike-sorting quality metrics."""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.sorting.metrics import (
    amplitude_drift,
    d_prime,
    d_prime_pairwise_matrix,
    fraction_missing,
    isolation_distance,
    l_ratio,
    peak_amplitude_snr,
    waveform_stability,
)

# Shared helpers from conftest.py (plain functions, not fixtures, so
# the existing call sites ``_two_clusters(sep=10.0)`` keep working).
from tests.conftest import make_two_clusters, make_overlapping_clusters, make_waveforms


def _two_clusters(n=200, sep=5.0, dim=10, rng_seed=894131):
    """Thin wrapper preserving the original call convention."""
    d = make_two_clusters(n=n, sep=sep, dim=dim, rng_seed=rng_seed)
    return d["features"], d["labels"]


def _overlapping_clusters(n=200, dim=10, rng_seed=89461):
    d = make_overlapping_clusters(n=n, dim=dim, rng_seed=rng_seed)
    return d["features"], d["labels"]


def _make_waveforms(n=300, snippet_len=38, rng_seed=42):
    d = make_waveforms(n=n, snippet_len=snippet_len, rng_seed=rng_seed)
    return d["waveforms"], d["labels"], d["template0"], d["template1"]


# ---------------------------------------------------------------------------
# Isolation distance
# ---------------------------------------------------------------------------

class TestIsolationDistance:
    def test_well_separated_high(self):
        X, lab = _two_clusters(sep=10.0)
        result = isolation_distance(X, lab)
        # Both clusters should have high isolation distance
        for v in result.values():
            assert v > 10, f"Expected high isolation, got {v}"

    def test_overlapping_low(self):
        X, lab = _overlapping_clusters()
        result = isolation_distance(X, lab)
        well_sep = isolation_distance(*_two_clusters(sep=10.0))
        # Overlapping should have much lower isolation distance than well-separated
        assert np.mean(list(result.values())) < np.mean(list(well_sep.values()))

    def test_single_cluster_returns_dict(self):
        X, lab = _two_clusters()
        val = isolation_distance(X, lab, all_clusters=False, cluster_id=0)
        assert isinstance(val, float)

    def test_too_few_spikes_nan(self):
        X = np.array([[1, 2]])
        lab = np.array([0])
        result = isolation_distance(X, lab)
        assert np.isnan(result[0])


# ---------------------------------------------------------------------------
# L-ratio
# ---------------------------------------------------------------------------

class TestLRatio:
    def test_well_separated_low(self):
        X, lab = _two_clusters(sep=10.0)
        result = l_ratio(X, lab)
        for v in result.values():
            assert v < 0.1, f"Expected low L-ratio, got {v}"

    def test_overlapping_higher(self):
        X, lab = _overlapping_clusters()
        result = l_ratio(X, lab)
        well_sep = l_ratio(*_two_clusters(sep=10.0))
        # Overlapping should have higher L-ratio on average
        assert np.mean(list(result.values())) > np.mean(list(well_sep.values()))


# ---------------------------------------------------------------------------
# d-prime
# ---------------------------------------------------------------------------

class TestDPrime:
    def test_scales_with_separation(self):
        X_close, lab = _two_clusters(sep=2.0)
        X_far, _ = _two_clusters(sep=10.0)
        dp_close = d_prime(X_close, lab)
        dp_far = d_prime(X_far, lab)
        # Far-apart clusters should have larger d'
        assert min(dp_far.values()) > min(dp_close.values())

    def test_single_cluster_nan(self):
        X = np.random.randn(50, 5)
        lab = np.zeros(50, dtype=int)
        result = d_prime(X, lab)
        assert np.isnan(result[0])

    def test_positive_values(self):
        X, lab = _two_clusters(sep=5.0)
        result = d_prime(X, lab)
        for v in result.values():
            assert v > 0

    # ------------------------------------------------------------------
    # Regression tests for the variance-formula bug
    # ------------------------------------------------------------------
    # The pre-fix implementation used ``np.var(Xc)`` (global variance of
    # the flattened cluster matrix), which equals
    # ``mean(per-feature var) + var(per-feature means)`` by the law of
    # total variance.  For waveform-shaped data the second term
    # dominates, so d-prime values were inflated denominators -> wrong
    # numbers.  These tests pin the value against the closed-form result.

    def test_regression_analytic_two_unit_gaussians(self):
        """Two isotropic unit-variance Gaussians, separation = 4.

        Closed form:
            σ²_A = σ²_B = 1
            pooled_std = sqrt(0.5 * (1 + 1)) = 1
            ||μ_A − μ_B||_2 = 4
            d' = 4 / 1 = 4
        With finite samples we tolerate ±0.15.
        """
        rng = np.random.default_rng(20260406)
        n, dim = 5_000, 8
        # Separation along the first axis only -> ||Δμ|| = 4 exactly
        X0 = rng.standard_normal((n, dim))
        delta = np.zeros(dim)
        delta[0] = 4.0
        X1 = rng.standard_normal((n, dim)) + delta
        X = np.vstack([X0, X1])
        lab = np.array([0] * n + [1] * n)
        result = d_prime(X, lab)
        assert result[0] == pytest.approx(4.0, abs=0.15)
        assert result[1] == pytest.approx(4.0, abs=0.15)

    def test_regression_per_feature_variance_not_global(self):
        """The bug surfaces when per-feature means span a wide range.

        We construct two clusters where the per-feature variance is
        ``1.0`` everywhere, but the per-feature means are
        ``[0, 1, 2, …, dim-1]`` for cluster A and the same shifted by
        ``+10`` along feature 0 for cluster B.

        Correct:
            σ² (per-dim mean of feature variances) = 1.0
            pooled_std = 1.0
            ||Δμ|| = 10
            d' = 10
        Buggy ``np.var(Xc)``:
            global var ≈ 1 + var([0..dim-1]) = 1 + dim²/12 (≈ 6.33 for
            dim=10), pooled_std ≈ √6.33 ≈ 2.52, d' ≈ 3.97
        The test rejects the buggy answer with a wide margin.
        """
        rng = np.random.default_rng(20260406)
        n, dim = 4_000, 10
        per_feature_mean = np.arange(dim, dtype=np.float64)
        X0 = rng.standard_normal((n, dim)) + per_feature_mean
        X1 = rng.standard_normal((n, dim)) + per_feature_mean
        X1[:, 0] += 10.0  # only feature 0 differs
        X = np.vstack([X0, X1])
        lab = np.array([0] * n + [1] * n)
        result = d_prime(X, lab)
        # Correct closed form is 10.0 ± a few percent
        assert result[0] == pytest.approx(10.0, abs=0.3)
        assert result[1] == pytest.approx(10.0, abs=0.3)
        # And it must be much larger than the buggy answer (~4)
        assert min(result.values()) > 7.0

    def test_pairwise_matrix_helper_symmetry(self):
        X, lab = _two_clusters(sep=4.0, n=1_000, dim=6)
        mat, ids = d_prime_pairwise_matrix(X, lab)
        assert mat.shape == (2, 2)
        assert np.array_equal(ids, np.array([0, 1]))
        # Diagonal NaN
        assert np.isnan(mat[0, 0])
        assert np.isnan(mat[1, 1])
        # Symmetric off-diagonal
        assert mat[0, 1] == pytest.approx(mat[1, 0], rel=1e-12)
        # Same minimum that d_prime() returns
        per = d_prime(X, lab)
        assert per[0] == pytest.approx(mat[0, 1], rel=1e-12)

    def test_zero_pooled_std_returns_nan(self):
        """Two single-row clusters give zero variance -> NaN, not inf."""
        X = np.array([[0.0, 0.0], [5.0, 5.0]])
        lab = np.array([0, 1])
        result = d_prime(X, lab)
        # Both clusters have len < 2 → undefined per-dim variance
        assert np.isnan(result[0])
        assert np.isnan(result[1])

    def test_plotting_uses_same_helper(self):
        """plot_d_prime_matrix must read identical numbers to d_prime."""
        # This guards against future drift between metrics.py and
        # plotting.py implementations.
        from neural_cca.sorting.plotting import (
            d_prime_pairwise_matrix as _from_plot,
        )
        X, lab = _two_clusters(sep=3.0, n=500, dim=8)
        mat_metric, _ = d_prime_pairwise_matrix(X, lab)
        mat_plot, _ = _from_plot(X, lab)
        np.testing.assert_array_equal(mat_metric, mat_plot)


# ---------------------------------------------------------------------------
# Peak amplitude SNR
# ---------------------------------------------------------------------------

class TestPeakAmplitudeSNR:
    def test_clean_signal_high_snr(self):
        wv, lab, _, _ = _make_waveforms()
        result = peak_amplitude_snr(wv, lab)
        for v in result.values():
            assert v > 3, f"Expected high SNR, got {v}"

    def test_noisy_signal_lower_snr(self):
        rng = np.random.default_rng(0)
        wv_noisy = rng.standard_normal((100, 38))
        val = peak_amplitude_snr(wv_noisy)
        assert val < 5  # pure noise ≈ low SNR


# ---------------------------------------------------------------------------
# Waveform stability
# ---------------------------------------------------------------------------

class TestWaveformStability:
    def test_stable_waveforms_high_r(self):
        wv, lab, _, _ = _make_waveforms()
        spike_times = np.arange(len(wv), dtype=np.float64) * 0.001
        result = waveform_stability(spike_times, wv, lab)
        for v in result.values():
            assert v > 0.95, f"Expected high stability, got {v}"

    def test_drifting_waveforms_lower_r(self):
        rng = np.random.default_rng(1)
        n = 500
        t = np.linspace(0, 1, 38)
        template = -5.0 * np.sin(np.pi * t)
        # Shape-changing drift: peak shifts position over time
        wv = np.empty((n, 38))
        for i in range(n):
            shift = int(10 * i / n)  # peak drifts by up to 10 samples
            wv[i] = np.roll(template, shift) + rng.standard_normal(38) * 0.3
        st = np.arange(n, dtype=np.float64) * 0.001
        val = waveform_stability(st, wv)
        # Shape drift → correlation well below 1
        assert val < 0.95, f"Expected lower stability with shape drift, got {val}"

    def test_too_few_spikes_nan(self):
        wv = np.random.randn(2, 38)
        st = np.array([0.0, 0.001])
        assert np.isnan(waveform_stability(st, wv))


# ---------------------------------------------------------------------------
# Amplitude drift
# ---------------------------------------------------------------------------

class TestAmplitudeDrift:
    def test_no_drift_near_zero(self):
        wv, lab, _, _ = _make_waveforms()
        result = amplitude_drift(wv, lab)
        for v in result.values():
            assert abs(v) < 0.3, f"Expected near-zero drift, got {v}"

    def test_systematic_drift_detected(self):
        rng = np.random.default_rng(2)
        n = 500
        t = np.linspace(0, 1, 38)
        template = -5.0 * np.sin(np.pi * t)
        # Growing amplitude
        scale = np.linspace(1, 3, n)[:, None]
        wv = template * scale + rng.standard_normal((n, 38)) * 0.1
        val = amplitude_drift(wv)
        assert val > 0.5, f"Expected positive drift, got {val}"


# ---------------------------------------------------------------------------
# Fraction missing
# ---------------------------------------------------------------------------

class TestFractionMissing:
    def test_well_detected_low_fraction(self):
        wv, lab, _, _ = _make_waveforms(n=500)
        result = fraction_missing(wv, lab)
        for v in result.values():
            assert v < 0.2, f"Expected low missing, got {v}"

    def test_few_spikes_nan(self):
        wv = np.random.randn(5, 38)
        assert np.isnan(fraction_missing(wv))

    def test_returns_between_0_and_1(self):
        wv, lab, _, _ = _make_waveforms()
        result = fraction_missing(wv, lab)
        for v in result.values():
            assert 0 <= v <= 1 or np.isnan(v)
