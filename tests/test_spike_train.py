"""Tests for spike train analysis functions."""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.sta.analysis import (
    autocorrelogram,
    cv_log_isi,
    fano_factor,
    firing_rate_stability,
    first_spike_latency,
    isi_violation_rate,
    local_variation,
    minimal_spike_train_analysis,
    psth,
    trial_to_trial_reliability,
)
from tests.conftest import (
    make_globally_sorted_poisson,
    make_identical_trials,
    make_poisson_spikes,
    make_regular_spikes,
)

# ---------------------------------------------------------------------------
# Thin wrappers preserving the (spike_times, trials) tuple convention
# ---------------------------------------------------------------------------


def _regular_spikes(rate=50.0, duration=2.5, n_trials=20, stim_onset=0.5):
    d = make_regular_spikes(rate=rate, duration=duration, n_trials=n_trials)
    return d["spike_times"], d["trials"]


def _poisson_spikes(rate=50.0, duration=2.5, n_trials=20, rng_seed=42):
    d = make_poisson_spikes(rate=rate, duration=duration, n_trials=n_trials, rng_seed=rng_seed)
    return d["spike_times"], d["trials"]


def _identical_trials(n_spikes_per_trial=10, duration=2.5, n_trials=20):
    d = make_identical_trials(
        n_spikes_per_trial=n_spikes_per_trial, duration=duration, n_trials=n_trials
    )
    return d["spike_times"], d["trials"]


# ---------------------------------------------------------------------------
# ISI violation rate
# ---------------------------------------------------------------------------


class TestISIViolationRate:
    def test_regular_no_violations(self):
        st, _ = _regular_spikes(rate=50.0)
        # ISI = 20 ms, refractory = 1 ms → no violations
        assert isi_violation_rate(st) == 0.0

    def test_violations_detected(self):
        st = np.array([0.0, 0.0005, 0.01, 0.0105, 0.02])
        rate = isi_violation_rate(st, refractory_period=0.001)
        assert rate > 0

    def test_empty_returns_zero(self):
        assert isi_violation_rate(np.array([0.5])) == 0.0

    # ------------------------------------------------------------------
    # Regression: trial-based data must use ``trials`` to avoid the
    # double-bug where (a) inter-trial spike pairs are counted as
    # refractory violations after a global sort, and (b) the rate is
    # divided by the trial-window size instead of the total recording
    # duration — together inflating a 4 Hz Poisson cell to ~600 Hz.
    # ------------------------------------------------------------------

    @staticmethod
    def _trial_based_poisson(rate_hz, n_trials=240, trial_dur=2.5, seed=42):
        d = make_globally_sorted_poisson(
            rate=rate_hz,
            n_trials=n_trials,
            trial_dur=trial_dur,
            rng_seed=seed,
        )
        return d["spike_times"], d["trials"]

    def test_trial_aware_poisson_rate_is_sane(self):
        spk, tr = self._trial_based_poisson(rate_hz=4.0)
        # Without trials → catastrophic over-count from the old bug.
        bad = isi_violation_rate(spk)
        assert bad > 100.0, (
            "Sanity check on the buggy continuous-mode behaviour: "
            "trial-relative spikes that are sorted globally must "
            "produce a wildly inflated rate (this guards the negative "
            "case so the test does not silently regress)."
        )
        # Trial-aware mode → expected rate is lambda^2 * tau ~ 0.016 Hz
        # for a 4 Hz Poisson cell with a 1 ms refractory period.
        good = isi_violation_rate(
            spk,
            trials=tr,
            trial_duration=2.5,
        )
        assert good < 0.1, (
            f"Trial-aware ISI violation rate must be near zero for a "
            f"clean 4 Hz Poisson cell, got {good} Hz"
        )

    def test_trial_aware_injected_violations(self):
        # 240 trials, one injected violation per trial → exactly 240
        # violations across (240 * 2.5) = 600 s ⇒ 0.4 Hz.
        rng = np.random.default_rng(0)
        n_trials, trial_dur = 240, 2.5
        spk, tr = [], []
        for t in range(n_trials):
            base = np.sort(rng.uniform(0, trial_dur, 10))
            base = np.append(base, base[-1] + 0.0005)
            spk.append(base)
            tr.append(np.full(len(base), t, dtype=np.int64))
        spk = np.concatenate(spk)
        tr = np.concatenate(tr)
        rate = isi_violation_rate(
            spk,
            trials=tr,
            trial_duration=trial_dur,
        )
        assert 0.35 < rate < 0.45, f"Expected 240 violations / 600 s ≈ 0.4 Hz, got {rate}"

    def test_trial_aware_requires_trial_duration(self):
        with pytest.raises(ValueError, match="trial_duration"):
            isi_violation_rate(
                np.array([0.1, 0.2]),
                trials=np.array([0, 0]),
            )


# ---------------------------------------------------------------------------
# Trial-relative bug class — every ISI / pair-based metric must be
# trial-aware on trial-relative data, otherwise cross-trial spike pairs
# leak in as fake ISIs and contaminate the result.  These tests cover
# every public function affected by the bug class so it can never
# silently regress.
# ---------------------------------------------------------------------------


def _globally_sorted_trial_poisson(
    rate=4.0,
    n_trials=240,
    trial_dur=2.5,
    seed=42,
):
    """Thin wrapper around conftest.make_globally_sorted_poisson."""
    d = make_globally_sorted_poisson(
        rate=rate,
        n_trials=n_trials,
        trial_dur=trial_dur,
        rng_seed=seed,
    )
    return d["spike_times"], d["trials"]


class TestTrialRelativeBugClass:
    def test_minimal_lvr_poisson_is_near_one_with_trials(self):
        spk, tr = _globally_sorted_trial_poisson()
        # Without trials → contaminated by cross-trial diffs.
        bad = minimal_spike_train_analysis(
            spk,
            n_trials=240,
            stim_window=(0.0, 2.5),
        )
        # With trials → real Poisson statistics.
        good = minimal_spike_train_analysis(
            spk,
            trials=tr,
            stim_window=(0.0, 2.5),
        )
        # Bug used to give LvR ≈ 4–5; correct value for Poisson is ~1.
        assert bad["lvr"] > 2.0, (
            "buggy LvR must reproduce so the test does not silently lose its negative case"
        )
        assert 0.7 < good["lvr"] < 1.3, f"trial-aware LvR must be ~1 for Poisson, got {good['lvr']}"
        # CV is closer to 1 in both modes for Poisson, but the trial-aware
        # value should be tighter.
        assert 0.85 < good["cv"] < 1.15, f"trial-aware CV ~1 for Poisson, got {good['cv']}"

    def test_local_variation_detects_bursts_only_with_trials(self):
        # Bursty cell: 5 bursts/trial, 3 spikes/burst at 500 Hz.  LV
        # should be high (>1.5) when computed correctly; the bug used
        # to mask the bursts and report LV ~1 (Poisson-like).
        rng = np.random.default_rng(0)
        n_trials, trial_dur = 100, 2.5
        spk, tr = [], []
        for t in range(n_trials):
            base = rng.uniform(0, trial_dur - 0.01, 5)
            burst = np.sort(np.concatenate([np.array([b, b + 0.002, b + 0.004]) for b in base]))
            spk.append(burst)
            tr.append(np.full(len(burst), t, dtype=np.int64))
        spk = np.concatenate(spk)
        tr = np.concatenate(tr)
        order = np.argsort(spk)
        spk, tr = spk[order], tr[order]

        lv_bad = local_variation(spk)
        lv_good = local_variation(spk, trials=tr)
        assert lv_bad < 1.2, (
            "buggy LV used to wash out bursts; reproduce so the test "
            "does not silently lose its negative case"
        )
        assert lv_good > 1.4, f"trial-aware LV must detect bursts, got {lv_good}"

    def test_cv_log_isi_with_trials_matches_per_trial_pool(self):
        spk, tr = _globally_sorted_trial_poisson(rate=10.0)
        # Compute the pooled-within-trial CV(log10 ISI) by hand.
        manual_isis = []
        for t in np.unique(tr):
            s = np.sort(spk[tr == t])
            if len(s) >= 2:
                manual_isis.append(np.diff(s))
        manual_isis = np.concatenate(manual_isis)
        log = np.log10(manual_isis)
        expected = float(np.std(log) / abs(np.mean(log)))

        got = cv_log_isi(spk, trials=tr)
        assert got == pytest.approx(expected, rel=1e-9)

    def test_autocorrelogram_total_count_scales_correctly(self):
        spk, tr = _globally_sorted_trial_poisson(rate=4.0)
        # 4 Hz Poisson: expected within-trial pair count per lag bin is
        # roughly lambda^2 * bin_size * total_duration.
        bin_size, max_lag = 0.001, 0.05
        n_trials_total = int(len(np.unique(tr)))
        expected_per_bin = (4.0**2) * bin_size * (n_trials_total * 2.5)

        bad_lags, bad_counts = autocorrelogram(
            spk,
            bin_size=bin_size,
            max_lag=max_lag,
        )
        good_lags, good_counts = autocorrelogram(
            spk,
            trials=tr,
            bin_size=bin_size,
            max_lag=max_lag,
        )
        # Bug used to put the inflated counts ~ n_trials × correct value.
        assert bad_counts.sum() > 50 * good_counts.sum(), (
            "buggy ACG must remain dramatically inflated so this test "
            "guards both the negative and the positive case"
        )
        # Correct ACG mean per bin should match the analytic prediction
        # within Poisson noise (~ sqrt(N) per bin).
        mean_per_bin = good_counts.mean()
        assert 0.5 * expected_per_bin < mean_per_bin < 1.5 * expected_per_bin, (
            f"ACG mean per bin {mean_per_bin:.1f} should be near "
            f"the analytic expectation {expected_per_bin:.1f}"
        )

    def test_firing_rate_stability_cv_uses_within_trial_isis(self):
        spk, tr = _globally_sorted_trial_poisson(rate=20.0)
        # Each window has dense Poisson ISIs; CV should be close to 1.
        # Without proper trial handling, the per-window CV would be
        # contaminated by cross-trial pairs and biased low.
        out = firing_rate_stability(
            spk,
            trials=tr,
            stat="cv",
            window_size=0.5,
            trial_duration=2.5,
        )
        # All windows should be defined and finite.
        assert np.all(np.isfinite(out["values"]))
        # Mean per-window CV should land in a reasonable Poisson range.
        # (Truncating ISIs to a window biases CV slightly below 1, but
        # not below 0.5.)
        assert 0.5 < out["mean"] < 1.3, f"per-window CV mean = {out['mean']:.3f}, expected ~0.7-1.0"


# ---------------------------------------------------------------------------
# Firing rate stability
# ---------------------------------------------------------------------------


class TestFiringRateStability:
    def test_constant_rate_low_cv(self):
        st, trials = _regular_spikes(rate=50.0)
        result = firing_rate_stability(st, trials, stat="mean", trial_duration=2.5)
        # Constant rate → CV of stat should be low
        assert result["cv_of_stat"] < 0.5

    def test_returns_expected_keys(self):
        st, trials = _poisson_spikes()
        result = firing_rate_stability(st, trials)
        assert "values" in result
        assert "mean" in result
        assert "std" in result
        assert "cv_of_stat" in result

    def test_unknown_stat_raises(self):
        st, trials = _regular_spikes()
        with pytest.raises(ValueError, match="Unknown stat"):
            firing_rate_stability(st, trials, stat="bogus")


# ---------------------------------------------------------------------------
# Autocorrelogram
# ---------------------------------------------------------------------------


class TestAutocorrelogram:
    def test_symmetric(self):
        st, _ = _poisson_spikes(rate=100.0, n_trials=5)
        lags, counts = autocorrelogram(st, bin_size=0.001, max_lag=0.02)
        # ACG should be roughly symmetric
        mid = len(counts) // 2
        left = counts[:mid]
        right = counts[mid + 1 :][::-1]
        # Trim to equal length in case of odd bin count
        min_len = min(len(left), len(right))
        left = left[:min_len]
        right = right[:min_len]
        # Allow some noise but overall shape should be similar
        corr = np.corrcoef(left.astype(float), right.astype(float))[0, 1]
        assert corr > 0.8, f"ACG not symmetric enough: r={corr}"

    def test_zero_lag_excluded(self):
        st = np.array([0.0, 0.01, 0.02, 0.03])
        lags, counts = autocorrelogram(st, bin_size=0.005, max_lag=0.04)
        mid = len(counts) // 2
        assert counts[mid] == 0, "Zero-lag bin should be 0"

    def test_rate_normalisation(self):
        """``normalize='rate'`` reports counts divided by n_spikes·bin_size."""
        rng = np.random.default_rng(7)
        st = np.sort(rng.uniform(0.0, 1.0, 50))
        bin_size = 0.001
        max_lag = 0.02
        _, counts = autocorrelogram(st, bin_size=bin_size, max_lag=max_lag, normalize="counts")
        _, rate = autocorrelogram(st, bin_size=bin_size, max_lag=max_lag, normalize="rate")
        # rate == counts / (n_spikes * bin_size) bin-for-bin
        expected = counts.astype(float) / (len(st) * bin_size)
        np.testing.assert_allclose(rate, expected, rtol=1e-12)
        assert rate.dtype == np.float64
        assert counts.dtype == np.int64

    def test_invalid_normalize_raises(self):
        with pytest.raises(ValueError, match="counts.*rate"):
            autocorrelogram(np.array([0.0, 0.001]), normalize="probability")  # type: ignore[arg-type]

    def test_empty_spike_train_rate_is_nan(self):
        """Empty input + ``normalize='rate'`` returns NaN, not a div-by-zero."""
        lags, rate = autocorrelogram(
            np.empty(0),
            bin_size=0.001,
            max_lag=0.01,
            normalize="rate",
        )
        assert rate.dtype == np.float64
        assert np.all(np.isnan(rate))
        assert lags.shape == rate.shape

    def test_plot_warns_on_misaligned_refractory(self):
        """plot_autocorrelogram warns when refr is not a multiple of bin_size.

        In that regime the dashed refractory line falls *inside* a
        bar rather than on a bin edge, so the visual "everything left
        of the line is a violation" reading breaks down.  The polish
        emits a RuntimeWarning so the user sees it instead of
        silently misreading the plot.
        """
        import matplotlib

        matplotlib.use("Agg")  # headless backend, no display required
        from neural_cca.sta.plotting import plot_autocorrelogram

        st = np.sort(np.random.default_rng(0).uniform(0.0, 1.0, 30))
        with pytest.warns(RuntimeWarning, match="whole multiple of bin_size"):
            plot_autocorrelogram(
                st,
                bin_size=0.001,
                max_lag=0.02,
                refractory_period=0.0015,  # 1.5 × bin_size
            )

    def test_plot_no_warning_when_aligned(self):
        """Aligned refractory (multiple of bin_size) → no warning."""
        import warnings as _warn

        import matplotlib

        matplotlib.use("Agg")
        from neural_cca.sta.plotting import plot_autocorrelogram

        st = np.sort(np.random.default_rng(0).uniform(0.0, 1.0, 30))
        with _warn.catch_warnings():
            _warn.simplefilter("error", RuntimeWarning)
            plot_autocorrelogram(
                st,
                bin_size=0.001,
                max_lag=0.02,
                refractory_period=0.002,  # 2 × bin_size, exact
            )

    def test_vectorised_matches_naive(self):
        """Regression test for the vectorised pair accumulation.

        Rewriting the inner pair loop to use ``np.searchsorted`` +
        batched ``np.histogram`` removes an O(n²) histogram-call hot
        spot.  This test pins the numerical equivalence against the
        naïve, per-pair implementation on a small Poisson train so
        any future rewrite is forced to remain bit-identical (or
        at least bin-identical).
        """
        rng = np.random.default_rng(123)
        # ~150 spikes drawn from a 100 Hz Poisson process across 1.5 s
        st = np.sort(rng.uniform(0.0, 1.5, 150))
        bin_size = 0.001
        max_lag = 0.05

        lags, counts = autocorrelogram(st, bin_size=bin_size, max_lag=max_lag)

        # Naïve reference: iterate every ordered pair, accumulate ±diff
        # via per-pair histograms.  Stays the same construction the
        # original implementation used so we can pin the new code to
        # the old number bin-for-bin.
        n_half = int(np.ceil(max_lag / bin_size))
        n_bins = 2 * n_half
        edges = np.linspace(-n_half * bin_size, n_half * bin_size, n_bins + 1)
        ref = np.zeros(n_bins, dtype=np.int64)
        eff_lag = n_half * bin_size
        for i in range(len(st)):
            for j in range(i + 1, len(st)):
                d = st[j] - st[i]
                if d > eff_lag:
                    break
                ref += np.histogram([d, -d], bins=edges)[0]

        np.testing.assert_array_equal(counts, ref)


# ---------------------------------------------------------------------------
# Fano factor
# ---------------------------------------------------------------------------


class TestFanoFactor:
    def test_regular_fano_low(self):
        st, trials = _regular_spikes(rate=50.0, n_trials=50)
        ff = fano_factor(st, trials, mode="per_trial")
        # Regular firing → very low Fano
        assert ff < 0.5, f"Expected Fano < 0.5 for regular, got {ff}"

    def test_poisson_fano_near_one(self):
        st, trials = _poisson_spikes(rate=50.0, n_trials=200, rng_seed=0)
        ff = fano_factor(st, trials, mode="per_trial")
        assert 0.5 < ff < 2.0, f"Expected Fano ≈ 1.0 for Poisson, got {ff}"

    def test_no_spikes_nan(self):
        ff = fano_factor(np.array([]), None, mode="per_bin")
        assert np.isnan(ff)

    def test_per_trial_requires_trials(self):
        import pytest as _pt

        with _pt.raises(ValueError, match="per_trial.*requires"):
            fano_factor(np.array([0.1, 0.2]), trials=None, mode="per_trial")

    def test_explicit_mode_no_warning(self):
        import warnings as _w

        st, trials = _poisson_spikes(rate=50.0, n_trials=20, rng_seed=0)
        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning becomes an error
            fano_factor(st, trials, mode="per_trial")

    def test_implicit_mode_emits_deprecation(self):
        import pytest as _pt

        st, trials = _poisson_spikes(rate=50.0, n_trials=20, rng_seed=0)
        with _pt.warns(DeprecationWarning, match="explicitly"):
            fano_factor(st, trials)


# ---------------------------------------------------------------------------
# Local variation (LV)
# ---------------------------------------------------------------------------


class TestLocalVariation:
    def test_regular_near_zero(self):
        st, _ = _regular_spikes(rate=50.0)
        lv = local_variation(st)
        assert lv < 0.1, f"Expected LV ≈ 0 for regular, got {lv}"

    def test_poisson_near_one(self):
        st, _ = _poisson_spikes(rate=50.0, n_trials=100, rng_seed=7)
        lv = local_variation(st)
        assert 0.5 < lv < 1.5, f"Expected LV ≈ 1 for Poisson, got {lv}"

    def test_few_spikes_nan(self):
        assert np.isnan(local_variation(np.array([0.5])))


# ---------------------------------------------------------------------------
# CV of log-ISI
# ---------------------------------------------------------------------------


class TestCvLogIsi:
    def test_regular_near_zero(self):
        st, _ = _regular_spikes(rate=50.0)
        val = cv_log_isi(st)
        # Constant ISI → log(ISI) constant → std ≈ 0
        assert val < 0.05, f"Expected logCV ≈ 0 for regular, got {val}"

    def test_poisson_positive(self):
        st, _ = _poisson_spikes(rate=50.0, n_trials=50, rng_seed=3)
        val = cv_log_isi(st)
        assert val > 0.1, f"Expected positive logCV for Poisson, got {val}"

    def test_few_spikes_nan(self):
        assert np.isnan(cv_log_isi(np.array([1.0])))


# ---------------------------------------------------------------------------
# PSTH
# ---------------------------------------------------------------------------


class TestPSTH:
    def test_shape(self):
        st, trials = _poisson_spikes(rate=50.0, duration=2.5, n_trials=10)
        centres, rate = psth(st, trials, bin_size=0.1, trial_duration=2.5)
        assert len(centres) == len(rate)
        assert len(centres) == 25  # 2.5 / 0.1

    def test_rate_reasonable(self):
        st, trials = _poisson_spikes(rate=50.0, duration=2.5, n_trials=100)
        _, rate = psth(st, trials, bin_size=0.5, trial_duration=2.5)
        # Mean rate should be near 50 Hz
        assert 20 < np.mean(rate) < 100

    def test_cluster_filtering(self):
        st, trials = _poisson_spikes(rate=50.0, n_trials=10)
        labels = np.zeros(len(st), dtype=int)
        labels[len(st) // 2 :] = 1
        _, rate0 = psth(st, trials, labels, 0, trial_duration=2.5)
        _, rate1 = psth(st, trials, labels, 1, trial_duration=2.5)
        # Each filtered cluster should produce a non-empty PSTH with
        # plausible rates (> 0 in at least some bins).  The rate values
        # themselves are normalised by different trial counts (each
        # cluster touches a different subset of trials), so comparing
        # sums directly is fragile.
        assert np.any(rate0 > 0)
        assert np.any(rate1 > 0)
        # The two spike sets are complementary: their raw spike counts
        # must partition the full set.
        n0 = int(np.sum(labels == 0))
        n1 = int(np.sum(labels == 1))
        assert n0 + n1 == len(st)
        assert n0 > 0 and n1 > 0


# ---------------------------------------------------------------------------
# Trial-to-trial reliability
# ---------------------------------------------------------------------------


class TestTrialReliability:
    def test_identical_trials_high(self):
        st, trials = _identical_trials(n_spikes_per_trial=20, n_trials=30)
        r = trial_to_trial_reliability(st, trials, stat="psth", bin_size=0.05, trial_duration=2.5)
        assert r > 0.9, f"Expected high reliability for identical trials, got {r}"

    def test_mfr_consistency(self):
        st, trials = _regular_spikes(rate=50.0, n_trials=30)
        r = trial_to_trial_reliability(st, trials, stat="mfr", trial_duration=2.5)
        # Constant rate → perfect consistency → score ≈ 1
        assert r > 0.9, f"Expected high MFR consistency, got {r}"

    def test_single_trial_nan(self):
        st = np.array([0.5, 1.0, 1.5])
        trials = np.array([0, 0, 0])
        assert np.isnan(trial_to_trial_reliability(st, trials))


# ---------------------------------------------------------------------------
# First spike latency
# ---------------------------------------------------------------------------


class TestFirstSpikeLatency:
    def test_known_latency(self):
        # Each trial has a spike 10 ms after stimulus onset (0.5 s)
        st = np.array([0.51, 0.51, 0.51])
        trials = np.array([0, 1, 2])
        result = first_spike_latency(st, trials, stim_onset=0.5)
        assert result["mean"] == pytest.approx(0.01, abs=1e-10)
        assert result["frac_responsive"] == 1.0

    def test_no_response_nan(self):
        # All spikes before stimulus onset
        st = np.array([0.1, 0.2, 0.3])
        trials = np.array([0, 1, 2])
        result = first_spike_latency(st, trials, stim_onset=0.5)
        assert np.isnan(result["mean"])
        assert result["frac_responsive"] == 0.0

    def test_partial_response(self):
        st = np.array([0.6, 0.1, 0.7])
        trials = np.array([0, 1, 2])
        result = first_spike_latency(st, trials, stim_onset=0.5)
        assert 0 < result["frac_responsive"] < 1

    def test_returns_expected_keys(self):
        st, trials = _poisson_spikes()
        result = first_spike_latency(st, trials)
        for key in ("latencies", "mean", "median", "std", "frac_responsive"):
            assert key in result
