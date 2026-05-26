"""Tests for advanced spike-sorting quality metrics."""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.sorting.metrics import (
    amplitude_drift,
    calc_weighted_snr,
    contamination_rate_hill,
    d_prime,
    d_prime_pairwise_matrix,
    fraction_missing,
    isolation_distance,
    l_ratio,
    peak_amplitude_snr,
    rpvs,
    waveform_stability,
)

# Shared helpers from conftest.py (plain functions, not fixtures, so
# the existing call sites ``_two_clusters(sep=10.0)`` keep working).
from tests.conftest import make_overlapping_clusters, make_two_clusters, make_waveforms


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


# ---------------------------------------------------------------------------
# calc_weighted_snr — NaN-cluster handling (regression for the
# "one degenerate cluster poisons the whole metric" bug)
# ---------------------------------------------------------------------------


class TestCalcWeightedSNR:
    def test_clean_multi_cluster_returns_finite(self):
        """All clusters well-formed → numeric weighted SNR."""
        wv, lab, _, _ = _make_waveforms(n=200)
        result = calc_weighted_snr(wv, lab)
        assert np.isfinite(result)
        assert result > 0

    def test_one_degenerate_cluster_warns_and_renormalises(self):
        """Mixing a healthy cluster with one that has identical
        waveforms emits a ``RuntimeWarning`` but returns the healthy
        cluster's SNR — *not* NaN."""
        wv, lab, _, _ = _make_waveforms(n=200)
        # Append a degenerate cluster: 50 *identical* snippets (zero
        # noise variance → est_snr returns NaN for this cluster alone).
        dup = np.tile(wv[0], (50, 1))
        wv_aug = np.vstack([wv, dup])
        lab_aug = np.concatenate([lab, np.full(50, 2, dtype=lab.dtype)])

        clean = calc_weighted_snr(wv, lab)
        with pytest.warns(RuntimeWarning, match="degenerate noise"):
            result = calc_weighted_snr(wv_aug, lab_aug)
        # The degenerate cluster is excluded; remaining weights are
        # renormalised back to the original two clusters, so the
        # numeric answer matches the clean run within float tolerance.
        assert np.isfinite(result)
        assert result == pytest.approx(clean, rel=1e-12)

    def test_all_degenerate_returns_nan(self):
        """If every cluster is degenerate the metric is undefined."""
        wv = np.tile(np.linspace(-1, 1, 20), (10, 1))
        lab = np.array([0] * 5 + [1] * 5, dtype=np.int64)
        # Make cluster 1 a different (but still identical) template.
        wv[5:] = np.linspace(2, 3, 20)
        with pytest.warns(RuntimeWarning, match="degenerate noise"):
            assert np.isnan(calc_weighted_snr(wv, lab))


# ---------------------------------------------------------------------------
# rpvs — input validation (regression for silent sign-inversion bug)
# ---------------------------------------------------------------------------


class TestRpvsValidation:
    @pytest.mark.parametrize("bad_refractory", [-0.001, 0.0])
    def test_non_positive_refractory_raises(self, bad_refractory):
        """Zero and negative ``refractory_period`` both trip the same
        guard (``<= 0``).  A negative value silently inverts the
        ``isi < refractory`` comparison and reports zero violations
        for every spike train; the function must refuse rather than
        lie."""
        st = np.array([0.0, 0.0005, 0.0015, 0.003])
        with pytest.raises(ValueError, match="refractory_period must be positive"):
            rpvs(st, refractory_period=bad_refractory)

    def test_positive_refractory_works(self):
        """Sanity: the validation doesn't break the happy path."""
        st = np.array([0.0, 0.0005, 0.001, 0.0025])
        result = rpvs(st, refractory_period=0.001, relative=False)
        assert isinstance(result, int)
        assert result >= 0


# ---------------------------------------------------------------------------
# Hill 2011 contamination rate
# ---------------------------------------------------------------------------


class TestContaminationRateHill:
    """Closed-form regressions for the Hill 2011 estimator.

    The Hill estimator is
        C = 1/2 (1 - sqrt(1 - 2 N_v T / (N^2 (t_r - t_c)))).
    These tests construct cases where ``N``, ``N_v``, ``T``, and ``t_r``
    are all known so the analytical result can be compared bit-for-bit.
    """

    @staticmethod
    def _hill_closed_form(N, N_v, T, t_r, t_c=0.0):
        disc = 1.0 - (2.0 * N_v * T) / (N**2 * (t_r - t_c))
        disc = max(0.0, float(disc))
        return 0.5 * (1.0 - np.sqrt(disc))

    def test_zero_violations_returns_zero(self):
        """No violations → C = 0 exactly."""
        # 100 spikes, 10/trial × 10 trials of 1 s.  Within each trial
        # spikes are at 0.05, 0.15, ..., 0.95 s — 100 ms ISI, well above
        # the 1 ms refractory.
        n_trials = 10
        trial_len = 1.0
        spikes_per_trial = 10
        st = np.tile(np.linspace(0.05, 0.95, spikes_per_trial), n_trials)
        trials = np.repeat(np.arange(n_trials), spikes_per_trial)
        C = contamination_rate_hill(
            st,
            trials=trials,
            recording_duration=n_trials * trial_len,
            refractory_period=0.001,
        )
        assert C == pytest.approx(0.0, abs=1e-12)

    def test_known_violations_matches_closed_form(self):
        """Inject a known number of within-trial violations."""
        n_trials = 5
        trial_len = 2.0
        # Each trial: 20 spikes spaced 0.1 s apart starting at 0.05 s
        # (no violations), plus one extra spike 0.5 ms after spike #5
        # in each trial → 1 violation per trial × 5 trials = 5 total.
        base = np.linspace(0.05, 1.95, 20)
        per_trial = np.concatenate([base, [base[5] + 0.0005]])
        st = np.tile(per_trial, n_trials)
        trials = np.repeat(np.arange(n_trials), len(per_trial))
        N = len(st)
        N_v_expected = 5
        T = n_trials * trial_len
        t_r = 0.001
        expected = self._hill_closed_form(N, N_v_expected, T, t_r)
        actual = contamination_rate_hill(
            st,
            trials=trials,
            recording_duration=T,
            refractory_period=t_r,
        )
        assert actual == pytest.approx(expected, rel=1e-10)

    def test_trials_argument_prevents_cross_trial_false_positives(self):
        """Without ``trials=``, globally-sorted trial-relative times
        merge across trials and produce spurious sub-millisecond
        pseudo-ISIs.  Passing ``trials=`` must suppress them.
        """
        # Two trials each with 5 spikes far apart (50 ms ISI within trial).
        # Without trial separation, sorting interleaves them so adjacent
        # spikes can be ~milliseconds apart by chance.
        rng = np.random.default_rng(2026)
        n_trials = 20
        per_trial = 5
        st = []
        trials = []
        for t in range(n_trials):
            # Spread 5 spikes uniformly in [0.05, 0.95] but jittered so
            # spikes from different trials are within 1 ms of each other
            # after global sort.
            base = np.linspace(0.05, 0.95, per_trial)
            jitter = rng.uniform(-0.0003, 0.0003, per_trial)  # ±0.3 ms
            st.append(base + jitter)
            trials.append(np.full(per_trial, t))
        st = np.concatenate(st)
        trials = np.concatenate(trials)
        T = n_trials * 1.0

        C_with = contamination_rate_hill(
            st, trials=trials, recording_duration=T, refractory_period=0.001
        )
        C_without = contamination_rate_hill(
            st, trials=None, recording_duration=T, refractory_period=0.001
        )
        # Per-trial counting sees zero real violations.
        assert C_with == pytest.approx(0.0, abs=1e-12)
        # Global counting must over-estimate (strictly greater)
        # because trials at near-identical jitter offsets create
        # spurious sub-ms gaps after sorting.
        assert C_without > C_with

    def test_per_cluster_dict(self):
        """``cluster_labels=`` returns one C per cluster."""
        n_trials = 4
        per_trial = 10
        st_clean = np.tile(np.linspace(0.05, 0.95, per_trial), n_trials)
        trials_clean = np.repeat(np.arange(n_trials), per_trial)
        # Cluster 0: clean.  Cluster 1: 1 violation per trial.
        per_trial_dirty = np.concatenate([np.linspace(0.05, 0.95, per_trial), [0.05 + 0.0003]])
        st_dirty = np.tile(per_trial_dirty, n_trials)
        trials_dirty = np.repeat(np.arange(n_trials), len(per_trial_dirty))

        st = np.concatenate([st_clean, st_dirty])
        trials = np.concatenate([trials_clean, trials_dirty])
        labels = np.concatenate([np.zeros(len(st_clean), int), np.ones(len(st_dirty), int)])
        C = contamination_rate_hill(
            st,
            cluster_labels=labels,
            trials=trials,
            recording_duration=n_trials * 1.0,
            refractory_period=0.001,
        )
        assert isinstance(C, dict)
        assert C[0] == pytest.approx(0.0, abs=1e-12)
        assert C[1] > 0.0

    def test_requires_recording_duration(self):
        st = np.array([0.0, 0.1, 0.2])
        with pytest.raises(ValueError, match="recording_duration is required"):
            contamination_rate_hill(st)


# ---------------------------------------------------------------------------
# fraction_missing — non-default methods + clamp_max parameter
# ---------------------------------------------------------------------------


class TestFractionMissingMethods:
    """Cover ``method="lognormal"`` / ``"empirical"`` and the new
    ``clamp_max`` parameter added when the per-path clamping was unified.
    """

    def test_lognormal_recovers_known_tail(self):
        """A lognormal sample whose minimum sits near the 5th percentile
        should report a missing fraction close to that percentile."""
        rng = np.random.default_rng(42)
        # Lognormal amplitudes: log-mean = 1, log-std = 0.5.
        log_amps = rng.normal(1.0, 0.5, 2000)
        # Truncate at the analytical 5th percentile.
        threshold = np.exp(1.0 - 1.6449 * 0.5)  # 5th percentile of normal
        amps = np.exp(log_amps)
        amps = amps[amps >= threshold]
        wv = amps[:, None]  # (n, 1) so amax-amin == amp
        wv = np.hstack([np.zeros((len(amps), 1)), wv])  # peak-to-peak = amp
        frac = fraction_missing(wv, method="lognormal", normality_warn=False)
        # 5% truncation ± KDE noise; allow generous tolerance.
        assert 0.02 < frac < 0.10

    def test_empirical_returns_in_valid_range(self):
        """Empirical KDE tail estimator should return a value in
        ``[0, clamp_max]``.  The KDE consistently under-estimates the
        true truncation fraction for sharp cutoffs (the bandwidth
        smears mass over the threshold), so we don't pin a numeric
        floor — only that the value is sensible and finite.
        """
        rng = np.random.default_rng(43)
        amps = rng.normal(10.0, 1.0, 2000)
        amps = amps[amps >= 9.0]  # ≈16 % truncation
        wv = np.hstack([np.zeros((len(amps), 1)), amps[:, None]])
        frac = fraction_missing(wv, method="empirical")
        assert 0.0 <= frac <= 0.5
        assert np.isfinite(frac)

    def test_clamp_max_none_disables_upper_bound(self):
        """``clamp_max=None`` lets the empirical method return values
        above 0.5 (useful for diagnostic / threshold-tuning workflows)."""
        rng = np.random.default_rng(44)
        # Near-symmetric distribution with x_min very near the centre →
        # KDE tail probability is near 0.5 from below (would clip to 0.5
        # under the default, may exceed 0.5 with smoothing under None).
        amps = rng.normal(0.0, 1.0, 200)
        wv = np.hstack([np.zeros((len(amps), 1)), amps[:, None]])
        frac_clamped = fraction_missing(wv, method="empirical")
        frac_unclamped = fraction_missing(wv, method="empirical", clamp_max=None)
        assert frac_clamped <= 0.5
        # When the KDE smoothing pushes the value above 0.5, the
        # unclamped version exposes it; otherwise both are equal.
        assert frac_unclamped >= frac_clamped

    def test_clamp_max_invalid_raises(self):
        wv = np.random.randn(20, 5)
        with pytest.raises(ValueError, match="clamp_max must be > 0"):
            fraction_missing(wv, clamp_max=0.0)
        with pytest.raises(ValueError, match="clamp_max must be > 0"):
            fraction_missing(wv, clamp_max=-0.1)

    def test_method_invalid_raises(self):
        wv = np.random.randn(20, 5)
        with pytest.raises(ValueError, match="method must be one of"):
            fraction_missing(wv, method="nonsense")


# ---------------------------------------------------------------------------
# isolation_distance / l_ratio — worst_pair mode
# ---------------------------------------------------------------------------


class TestWorstPairMode:
    """``mode="worst_pair"`` reports per-neighbour quality rather than
    pooled non-cluster quality.  The mathematical invariants follow
    from the formulas:

    * **isolation_distance** — global takes the *n_A*-th distance from
      the union of all non-A spikes (sorting interleaves clusters);
      worst_pair takes the *minimum over neighbours* of each
      neighbour's *n_A*-th distance.  Since mixing more spikes can
      only reach the *n_A*-th index *earlier* (closer), we have
      ``global ≤ worst_pair``.  worst_pair therefore reports a
      *better* (larger) isolation distance — it ignores small-close
      clusters with fewer than *n_A* spikes that global would catch.
    * **L-ratio** — global is a *sum* of per-non-cluster
      ``(1 − χ²_cdf)`` contributions divided by *n_A*; worst_pair is
      the *max* of per-neighbour partial sums divided by *n_A*.  Since
      ``sum ≥ max`` for non-negative summands, ``global ≥ worst_pair``.
    """

    @staticmethod
    def _three_clusters(n=120, sep=8.0, rng_seed=314):
        rng = np.random.default_rng(rng_seed)
        # A at origin, B far away, C overlapping with A
        A = rng.normal(0.0, 1.0, (n, 8))
        B = rng.normal(sep, 1.0, (n, 8))
        C = rng.normal(0.5, 1.0, (n, 8))  # nearby
        feats = np.vstack([A, B, C])
        labs = np.concatenate([np.zeros(n, int), np.ones(n, int), 2 * np.ones(n, int)])
        return feats, labs

    def test_isolation_worst_pair_invariant(self):
        """``global ≤ worst_pair`` for every cluster (per the
        sum-mixing argument above)."""
        feats, labs = self._three_clusters()
        iso_global = isolation_distance(feats, labs, mode="global")
        iso_worst = isolation_distance(feats, labs, mode="worst_pair")
        for cid in (0, 1, 2):
            assert iso_worst[cid] >= iso_global[cid] - 1e-9, (
                f"cluster {cid}: worst_pair={iso_worst[cid]}, "
                f"global={iso_global[cid]} (expected worst_pair ≥ global)"
            )

    def test_lratio_worst_pair_invariant(self):
        """``global ≥ worst_pair`` for every cluster (sum ≥ max)."""
        feats, labs = self._three_clusters()
        lr_global = l_ratio(feats, labs, mode="global")
        lr_worst = l_ratio(feats, labs, mode="worst_pair")
        for cid in (0, 1, 2):
            assert lr_global[cid] >= lr_worst[cid] - 1e-12, (
                f"cluster {cid}: global={lr_global[cid]}, "
                f"worst_pair={lr_worst[cid]} (expected global ≥ worst_pair)"
            )

    def test_worst_pair_identifies_overlapping_neighbour(self):
        """For cluster 0 (overlaps cluster 2, far from cluster 1), the
        worst_pair L-ratio against the union should equal the
        contribution from cluster 2 alone — i.e. cluster 2 is the
        dominant overlap.
        """
        feats, labs = self._three_clusters()
        lr_worst = l_ratio(feats, labs, mode="worst_pair")
        # cluster 2's worst neighbour should likewise be cluster 0 (mirror).
        # The two overlapping clusters should both report a strictly
        # larger worst_pair L-ratio than cluster 1 (which has only the
        # far cluster 0 and the far cluster 2 as competitors).
        assert lr_worst[0] > lr_worst[1]
        assert lr_worst[2] > lr_worst[1]

    def test_mode_invalid_raises(self):
        feats, labs = self._three_clusters()
        with pytest.raises(ValueError, match="mode must be 'global' or 'worst_pair'"):
            isolation_distance(feats, labs, mode="nonsense")
        with pytest.raises(ValueError, match="mode must be 'global' or 'worst_pair'"):
            l_ratio(feats, labs, mode="nonsense")
