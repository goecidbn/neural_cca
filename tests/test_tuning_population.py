"""Tests for population-level tuning analysis functions.

Run with:
    python -m pytest tests/test_tuning_population.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.tuning.population import (
    orientation_map_statistics,
    signal_correlations,
    noise_correlations,
)


# ======================================================================
# Tests: orientation_map_statistics
# ======================================================================

class TestOrientationMapStatistics:

    def test_uniform_distribution(self):
        """Uniformly distributed orientations → high Rayleigh p."""
        rng = np.random.default_rng(42)
        pref_oris = rng.uniform(0, 180, 100)
        result = orientation_map_statistics(pref_oris)
        assert result["rayleigh_p"] > 0.05, (
            f"Uniform dist: expected p > 0.05, got {result['rayleigh_p']:.3f}"
        )
        assert result["is_uniform"] is True

    def test_clustered_distribution(self):
        """Clustered orientations → low Rayleigh p."""
        rng = np.random.default_rng(42)
        # All neurons prefer ~90° ± 5°
        pref_oris = rng.normal(90, 5, 50) % 180
        result = orientation_map_statistics(pref_oris)
        assert result["rayleigh_p"] < 0.05, (
            f"Clustered: expected p < 0.05, got {result['rayleigh_p']:.3f}"
        )
        assert result["is_uniform"] is False

    def test_mean_ori_near_cluster(self):
        """Mean orientation should be near the cluster centre."""
        rng = np.random.default_rng(42)
        pref_oris = rng.normal(45, 3, 100) % 180
        result = orientation_map_statistics(pref_oris)
        assert abs(result["mean_ori"] - 45) < 10, (
            f"Expected mean ≈ 45°, got {result['mean_ori']:.1f}°"
        )

    def test_empty_input(self):
        """Empty array should return NaN values."""
        result = orientation_map_statistics(np.array([]))
        assert np.isnan(result["mean_ori"])
        assert result["is_uniform"] is True

    def test_concentration_range(self):
        """Concentration should be in [0, 1]."""
        rng = np.random.default_rng(42)
        pref_oris = rng.uniform(0, 180, 50)
        result = orientation_map_statistics(pref_oris)
        assert 0 <= result["concentration"] <= 1


# ======================================================================
# Tests: signal_correlations
# ======================================================================

class TestSignalCorrelations:

    def test_identical_tuning(self):
        """Identical tuning curves → r = 1."""
        tc = np.array([
            [10, 5, 2, 5, 10, 5, 2, 5],
            [10, 5, 2, 5, 10, 5, 2, 5],
        ], dtype=float)
        corr = signal_correlations(tc)
        assert corr[0, 1] == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_tuning(self):
        """Orthogonal tuning → negative signal correlation."""
        # Neuron 1: prefers 0°, Neuron 2: prefers 90°
        angles = np.linspace(0, 360, 8, endpoint=False)
        tc1 = 2.0 + 10.0 * np.exp(-((angles - 0) ** 2) / (2 * 30 ** 2))
        tc2 = 2.0 + 10.0 * np.exp(-((angles - 90) ** 2) / (2 * 30 ** 2))
        tc = np.vstack([tc1, tc2])
        corr = signal_correlations(tc)
        assert corr[0, 1] < 0.3  # Should be negative or weakly positive

    def test_diagonal_is_one(self):
        """Diagonal should be 1."""
        tc = np.random.default_rng(42).uniform(1, 10, (5, 8))
        corr = signal_correlations(tc)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(5))

    def test_symmetric(self):
        """Matrix should be symmetric."""
        tc = np.random.default_rng(42).uniform(1, 10, (4, 8))
        corr = signal_correlations(tc)
        np.testing.assert_array_almost_equal(corr, corr.T)


# ======================================================================
# Tests: noise_correlations
# ======================================================================

class TestNoiseCorrelations:

    def test_independent_noise(self):
        """Independent noise → correlations near 0."""
        rng = np.random.default_rng(42)
        n_neurons, n_trials = 3, 100
        angles = np.tile(np.linspace(0, 315, 8), n_trials // 8 + 1)[:n_trials]
        # Independent Poisson rates
        trial_rates = rng.poisson(10, (n_neurons, n_trials)).astype(float)
        corr = noise_correlations(trial_rates, angles)
        # Off-diagonal should be near 0
        mask = ~np.eye(n_neurons, dtype=bool)
        assert np.abs(corr[mask]).mean() < 0.2

    def test_correlated_noise(self):
        """Shared noise → positive correlations."""
        rng = np.random.default_rng(42)
        n_trials = 200
        angles = np.tile(np.linspace(0, 315, 8), n_trials // 8 + 1)[:n_trials]
        shared = rng.normal(0, 3, n_trials)
        r1 = 10 + shared + rng.normal(0, 1, n_trials)
        r2 = 10 + shared + rng.normal(0, 1, n_trials)
        trial_rates = np.vstack([r1, r2])
        corr = noise_correlations(trial_rates, angles)
        assert corr[0, 1] > 0.2, f"Expected positive r, got {corr[0, 1]:.3f}"

    def test_symmetric(self):
        """Matrix should be symmetric."""
        rng = np.random.default_rng(42)
        trial_rates = rng.uniform(5, 15, (3, 60))
        angles = np.tile(np.arange(8) * 45.0, 60 // 8 + 1)[:60]
        corr = noise_correlations(trial_rates, angles)
        np.testing.assert_array_almost_equal(corr, corr.T)
