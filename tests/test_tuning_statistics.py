"""Tests for statistical testing functions in tuning analysis.

Run with:
    python -m pytest tests/test_tuning_statistics.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.tuning.statistics import (
    orientation_selectivity_significance,
    anova_across_orientations,
    bootstrap_ci,
    bootstrap_ci_strata,
)
from neural_cca.tuning.selectivity import (
    dosi_circular_normalised,
)

# Shared helper from conftest.py
from tests.conftest import make_tuned_spikes as _make_tuned_spikes


# ======================================================================
# Tests: orientation_selectivity_significance
# ======================================================================

class TestOrientationSelectivitySignificance:

    def test_tuned_significant(self):
        """Strongly tuned neuron → significant OSI (orientation-symmetric)."""
        # OSI doubles angles, so strong at 90° AND 270° (=same orientation),
        # weak at 0° and 180°.  This creates high OSI that's hard to
        # replicate by random permutation.
        angles = np.linspace(0, 360, 12, endpoint=False)
        resp = np.array([1, 1, 1, 50, 50, 1, 1, 1, 1, 50, 50, 1], dtype=float)
        result = orientation_selectivity_significance(
            resp, angles, n_permutations=1000, rng=42,
        )
        assert result["p_permutation"] < 0.05, (
            f"Expected p < 0.05, got {result['p_permutation']}"
        )
        assert result["osi"] > 0.2

    def test_untuned_not_significant(self):
        """Flat response → not significant."""
        rng = np.random.default_rng(42)
        angles = np.linspace(0, 360, 12, endpoint=False)
        resp = 10.0 + rng.normal(0, 0.1, 12)  # tiny noise
        result = orientation_selectivity_significance(
            resp, angles, n_permutations=500, rng=42,
        )
        assert result["p_permutation"] > 0.05
        assert result["is_significant"] is False

    def test_returns_all_keys(self):
        """Result should have expected keys."""
        angles = np.linspace(0, 360, 8, endpoint=False)
        resp = np.array([20, 5, 2, 5, 20, 5, 2, 5], dtype=float)
        result = orientation_selectivity_significance(resp, angles, n_permutations=100)
        for key in ["osi", "p_permutation", "p_rayleigh", "is_significant"]:
            assert key in result


# ======================================================================
# Tests: anova_across_orientations
# ======================================================================

class TestAnovaAcrossOrientations:

    def test_tuned_significant_anova(self):
        """Tuned neuron → significant ANOVA."""
        st, tr, angles, _ = _make_tuned_spikes(
            preferred_angle=90.0, sigma_deg=25.0,
            peak_rate=30.0, base_rate=2.0,
        )
        result = anova_across_orientations(
            st, tr, angles, stim_window=(0.5, 2.5),
        )
        assert result["p_value"] < 0.05, (
            f"Expected ANOVA p < 0.05, got {result['p_value']:.4f}"
        )
        assert result["f_stat"] > 1.0

    def test_untuned_non_significant(self):
        """Untuned neuron → non-significant ANOVA."""
        st, tr, angles, _ = _make_tuned_spikes(
            preferred_angle=0.0, sigma_deg=9999.0,
            peak_rate=10.0, base_rate=10.0,
        )
        result = anova_across_orientations(
            st, tr, angles, stim_window=(0.5, 2.5),
        )
        assert result["p_value"] > 0.01, (
            f"Expected ANOVA p > 0.01, got {result['p_value']:.4f}"
        )

    def test_group_means_keys(self):
        """group_means should have one entry per unique angle."""
        st, tr, angles, angles_deg = _make_tuned_spikes(n_angles=8)
        result = anova_across_orientations(
            st, tr, angles, stim_window=(0.5, 2.5),
        )
        assert len(result["group_means"]) == 8
        assert len(result["group_stds"]) == 8


# ======================================================================
# Tests: bootstrap_ci
# ======================================================================

class TestBootstrapCI:

    def test_known_mean(self):
        """Bootstrap CI should bracket the true mean."""
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 200)
        result = bootstrap_ci(data, np.mean, n_bootstrap=500, rng=42)
        assert result["ci_lower"] < 5.0 < result["ci_upper"]
        assert result["estimate"] == pytest.approx(np.mean(data))

    def test_width_shrinks_with_n(self):
        """Larger samples → narrower CI."""
        rng = np.random.default_rng(42)
        small = rng.normal(0, 1, 20)
        large = rng.normal(0, 1, 500)

        ci_small = bootstrap_ci(small, np.mean, n_bootstrap=500, rng=42)
        ci_large = bootstrap_ci(large, np.mean, n_bootstrap=500, rng=42)

        width_small = ci_small["ci_upper"] - ci_small["ci_lower"]
        width_large = ci_large["ci_upper"] - ci_large["ci_lower"]
        assert width_large < width_small

    def test_custom_stat(self):
        """Should work with custom stat function."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = bootstrap_ci(data, np.std, n_bootstrap=200, rng=42)
        assert "estimate" in result
        assert result["se"] > 0

    def test_returns_all_keys(self):
        """Result should contain all expected keys."""
        data = np.arange(20, dtype=float)
        result = bootstrap_ci(data, np.mean, n_bootstrap=100)
        for key in ["estimate", "ci_lower", "ci_upper", "se"]:
            assert key in result


# ======================================================================
# Tests: bootstrap_ci_strata
# ======================================================================

class TestBootstrapCIStrata:
    """Tests for stratified bootstrap with paired (data, label) input."""

    def test_returns_all_keys(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=40)
        strata = np.repeat([0, 1, 2, 3], 10)
        result = bootstrap_ci_strata(
            data, strata, lambda d, s: float(np.mean(d)),
            n_bootstrap=100, rng=0,
        )
        for key in ("estimate", "ci_lower", "ci_upper", "se"):
            assert key in result

    def test_preserves_pairing_for_osi(self):
        """Stratified bootstrap should give a tight CI bracketing
        the true OSI of a tuned response."""
        # Strong orientation tuning, 12 angles, 20 trials per angle
        rng = np.random.default_rng(42)
        angles_unique = np.linspace(0, 360, 12, endpoint=False)
        n_trials_per = 20
        angles = np.repeat(angles_unique, n_trials_per)
        # Tuning curve: peak at 90° (and 270° because OSI doubles)
        true_rates = 2.0 + 18.0 * (
            np.cos(np.deg2rad(2 * (angles_unique - 90))) + 1.0
        ) / 2.0
        rates_per_angle = np.repeat(true_rates, n_trials_per)
        # Add Poisson-like noise
        data = rates_per_angle + rng.normal(0, 0.5, size=len(angles))

        result = bootstrap_ci_strata(
            data, angles,
            lambda d, s: dosi_circular_normalised(d, s),
            n_bootstrap=200, rng=42,
        )
        # The point estimate should be a meaningful OSI (>0.1)...
        assert result["estimate"] > 0.1
        # ...and the CI should bracket it.
        assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]
        # CI should be reasonably tight (not 0..1).
        assert (result["ci_upper"] - result["ci_lower"]) < 0.4

    def test_resampling_within_strata_only(self):
        """Verify that values from one stratum never end up in
        positions belonging to a different stratum."""
        n_per = 5
        strata = np.repeat([10, 20, 30, 40], n_per)
        # data values encode their stratum: 10..14 for stratum 10, etc.
        data = np.concatenate(
            [np.arange(s, s + n_per, dtype=float) for s in (10, 20, 30, 40)]
        )

        captured = {"resampled": None}

        def stat(d: np.ndarray, s: np.ndarray) -> float:
            captured["resampled"] = d.copy()
            return 0.0

        bootstrap_ci_strata(data, strata, stat, n_bootstrap=1, rng=7)

        resampled = captured["resampled"]
        # For each stratum group, every value in `resampled` at those
        # positions must come from the same stratum's value range.
        for s, lo in zip([10, 20, 30, 40], [10, 20, 30, 40]):
            mask = strata == s
            assert np.all((resampled[mask] >= lo) & (resampled[mask] < lo + n_per)), (
                f"Stratum {s} got cross-stratum values: {resampled[mask]}"
            )

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            bootstrap_ci_strata(
                np.arange(10, dtype=float),
                np.arange(8),
                lambda d, s: 0.0,
            )
