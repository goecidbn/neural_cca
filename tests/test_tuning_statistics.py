"""Tests for statistical testing functions in tuning analysis.

Run with:
    python -m pytest tests/test_tuning_statistics.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.tuning.selectivity import (
    dosi_circular_normalised,
)
from neural_cca.tuning.statistics import (
    anova_across_orientations,
    bootstrap_ci,
    bootstrap_ci_strata,
    orientation_selectivity_significance,
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
            resp,
            angles,
            n_permutations=1000,
            rng=42,
        )
        assert result["p_permutation"] < 0.05, f"Expected p < 0.05, got {result['p_permutation']}"
        assert result["osi"] > 0.2

    def test_untuned_not_significant(self):
        """Flat response → not significant."""
        rng = np.random.default_rng(42)
        angles = np.linspace(0, 360, 12, endpoint=False)
        resp = 10.0 + rng.normal(0, 0.1, 12)  # tiny noise
        result = orientation_selectivity_significance(
            resp,
            angles,
            n_permutations=500,
            rng=42,
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
            preferred_angle=90.0,
            sigma_deg=25.0,
            peak_rate=30.0,
            base_rate=2.0,
        )
        result = anova_across_orientations(
            st,
            tr,
            angles,
            stim_window=(0.5, 2.5),
        )
        assert result["p_value"] < 0.05, f"Expected ANOVA p < 0.05, got {result['p_value']:.4f}"
        assert result["f_stat"] > 1.0

    def test_untuned_non_significant(self):
        """Untuned neuron → non-significant ANOVA."""
        st, tr, angles, _ = _make_tuned_spikes(
            preferred_angle=0.0,
            sigma_deg=9999.0,
            peak_rate=10.0,
            base_rate=10.0,
        )
        result = anova_across_orientations(
            st,
            tr,
            angles,
            stim_window=(0.5, 2.5),
        )
        assert result["p_value"] > 0.01, f"Expected ANOVA p > 0.01, got {result['p_value']:.4f}"

    def test_group_means_keys(self):
        """group_means should have one entry per unique angle."""
        st, tr, angles, angles_deg = _make_tuned_spikes(n_angles=8)
        result = anova_across_orientations(
            st,
            tr,
            angles,
            stim_window=(0.5, 2.5),
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

    def test_default_method_is_bca(self):
        """``bootstrap_ci`` defaults to BCa as of v0.1.3."""
        data = np.arange(50, dtype=float)
        result = bootstrap_ci(data, np.mean, n_bootstrap=500, rng=42)
        assert "method" in result
        assert result["method"] == "bca"

    def test_bca_differs_from_percentile_on_skewed_data(self):
        """BCa applies a bias + acceleration correction; on a skewed
        sample its endpoints should differ from the plain percentile
        method.  We use a heavily right-skewed exponential.
        """
        rng = np.random.default_rng(7)
        data = rng.exponential(scale=2.0, size=300)
        bca = bootstrap_ci(data, np.mean, n_bootstrap=1000, rng=42, method="bca")
        pct = bootstrap_ci(data, np.mean, n_bootstrap=1000, rng=42, method="percentile")
        assert bca["method"] == "bca"
        assert pct["method"] == "percentile"
        # Both must bracket the true mean (= 2.0) at this sample size.
        assert bca["ci_lower"] < 2.0 < bca["ci_upper"]
        assert pct["ci_lower"] < 2.0 < pct["ci_upper"]
        # BCa endpoints should differ from percentile on a skewed sample.
        # A tiny tolerance handles the rare exact-match case.
        assert (
            abs(bca["ci_lower"] - pct["ci_lower"]) > 1e-6
            or abs(bca["ci_upper"] - pct["ci_upper"]) > 1e-6
        )

    def test_bca_falls_back_to_percentile_on_degenerate(self):
        """When the bootstrap distribution is degenerate (constant
        stat ⇒ z0 = ±inf), the BCa path should fall back to plain
        percentile rather than raise or return inf endpoints.
        """
        # Constant data → every bootstrap statistic is the same value,
        # so the BCa bias correction is degenerate.
        data = np.full(20, 3.0)
        result = bootstrap_ci(data, np.mean, n_bootstrap=200, rng=42, method="bca")
        assert result["method"] == "percentile"
        assert result["ci_lower"] == pytest.approx(3.0)
        assert result["ci_upper"] == pytest.approx(3.0)


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
            data,
            strata,
            lambda d, s: float(np.mean(d)),
            n_bootstrap=100,
            rng=0,
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
        true_rates = 2.0 + 18.0 * (np.cos(np.deg2rad(2 * (angles_unique - 90))) + 1.0) / 2.0
        rates_per_angle = np.repeat(true_rates, n_trials_per)
        # Add Poisson-like noise
        data = rates_per_angle + rng.normal(0, 0.5, size=len(angles))

        result = bootstrap_ci_strata(
            data,
            angles,
            lambda d, s: dosi_circular_normalised(d, s),
            n_bootstrap=200,
            rng=42,
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
        data = np.concatenate([np.arange(s, s + n_per, dtype=float) for s in (10, 20, 30, 40)])

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


# ======================================================================
# Tests: evaluate_os_per_cluster shares its rng across clusters
# ======================================================================


class TestEvaluateOsPerClusterRng:
    """Regression tests for the rng materialisation fix.

    Before the fix, an integer ``rng`` argument was forwarded raw to
    every ``get_os_metrics`` call, and each call rebuilt a fresh
    ``Generator`` from the same seed.  That made the bootstrap streams
    in different clusters **identical**, defeating the point of
    "share one stream across clusters" described in the docstring.

    The fix materialises a single ``Generator`` at the top of
    ``evaluate_os_per_cluster`` (via ``make_rng``) and reuses it.
    """

    def test_integer_seed_advances_across_clusters(self):
        from neural_cca.sorting.containers import SortingData
        from neural_cca.sorting.sorting import evaluate_os_per_cluster

        st, tr, angles, _ = _make_tuned_spikes(preferred_angle=90.0, sigma_deg=25.0)
        # Two synthetic clusters so the loop runs twice.
        cluster_labels = np.zeros(len(st), dtype=np.int64)
        cluster_labels[len(st) // 2 :] = 1

        # Match _make_tuned_spikes defaults so the SortingData container
        # has consistent shape: waveforms are unused for OS metrics but
        # must satisfy the n_spikes invariant.
        data = SortingData(
            waveforms=np.zeros((len(st), 8), dtype=np.float64),
            spike_times=st.astype(np.float64),
            trials=tr.astype(np.int64),
            angles=angles.astype(np.float64),
            n_trials=len(angles),
            stim_window=(0.5, 2.5),
            stim_frequency=2.0,
        )
        # Patch get_os_metrics and capture the Generator passed in for
        # each cluster.  If the fix is in place, both calls receive the
        # *same* Generator object that has been advanced between calls.
        from unittest.mock import patch

        from neural_cca.sorting import sorting as _sorting_mod

        captured: list[object] = []

        real = _sorting_mod.get_os_metrics

        def fake(*args, **kwargs):
            captured.append(kwargs.get("rng"))
            return real(*args, **kwargs)

        with patch.object(_sorting_mod, "get_os_metrics", side_effect=fake):
            evaluate_os_per_cluster(data, cluster_labels, rng=7)

        assert len(captured) == 2
        # Both calls saw a Generator (not the int 7) and it is the
        # *same* Generator instance — proving the rng was materialised
        # once and shared across clusters.
        assert isinstance(captured[0], np.random.Generator)
        assert captured[0] is captured[1]


# ======================================================================
# Tests: _build_trial_filter rejects bad trial IDs
# ======================================================================


class TestTrialIndexValidation:
    """The trial-index contract is now validated explicitly.

    ``angles[k]`` is the stimulus angle of trial ``k``, so every value
    in ``trials`` must be a valid index into ``angles``.  Out-of-range
    trial IDs used to silently mis-map angles; they now raise.
    """

    def test_out_of_range_trial_raises(self):
        from neural_cca.tuning._filter import _build_trial_filter

        # 5 angles → valid trial IDs are 0..4.  Trial 7 is out of range.
        with pytest.raises(ValueError, match=r"\[0, len\(angles\)\)"):
            _build_trial_filter(
                spike_times=np.array([0.7, 1.2]),
                trials=np.array([0, 7]),
                angles=np.linspace(0, 144, 5),
            )

    def test_negative_trial_raises(self):
        from neural_cca.tuning._filter import _build_trial_filter

        with pytest.raises(ValueError, match=r"\[0, len\(angles\)\)"):
            _build_trial_filter(
                spike_times=np.array([0.7, 1.2]),
                trials=np.array([-1, 0]),
                angles=np.linspace(0, 144, 5),
            )

    def test_in_range_trials_ok(self):
        from neural_cca.tuning._filter import _build_trial_filter

        # All trial IDs valid — should not raise.
        out = _build_trial_filter(
            spike_times=np.array([0.7, 1.2, 2.1]),
            trials=np.array([0, 1, 4]),
            angles=np.linspace(0, 144, 5),
        )
        assert out.n_trials == 5
        # Trials 2 and 3 have no spikes, so their MFR is 0.
        assert out.mfrs[2] == 0.0 and out.mfrs[3] == 0.0


# ======================================================================
# Tests: dosi_circular_normalised int-shorthand validation
# ======================================================================


class TestDosiIntShorthand:
    def test_int_shorthand_must_match_activities(self):
        with pytest.raises(ValueError, match="must equal len"):
            dosi_circular_normalised(np.ones(10), 8)

    def test_none_defaults_to_activity_length(self):
        # ``angles=None`` is equivalent to ``angles=len(activities)``,
        # so a flat 12-bin tuning curve yields OSI ≈ 0.
        val = dosi_circular_normalised(np.ones(12))
        assert val < 1e-12

    def test_explicit_matching_int_still_works(self):
        # Symmetric orientation profile across 8 evenly-spaced angles.
        activities = np.array([20, 5, 1, 5, 20, 5, 1, 5], dtype=float)
        val = dosi_circular_normalised(activities, 8)
        assert val > 0.3
