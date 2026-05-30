"""Tests for :mod:`neural_cca.synthetic`.

The synthetic module is the single source of truth for the spike-train
and waveform data used by the example notebooks and many of the
conftest fixtures.  Before this file existed, none of its three public
functions had direct tests — the audit cycle that introduced these
checks documents the rationale in ``docs/changelog.md``.

Run with::

    python -m pytest tests/test_synthetic.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.synthetic import (
    TwoUnitDemo,
    make_tuned_spikes,
    make_two_unit_demo,
    poisson_train,
)
from neural_cca.tuning.selectivity import gosi

# ======================================================================
# poisson_train
# ======================================================================


class TestPoissonTrain:
    def test_returns_int64_bin_indices(self):
        rng = np.random.default_rng(0)
        bins = poisson_train(rng, rate_profile=5.0, dt=1e-3, n_bins=2_500)
        assert bins.dtype == np.int64
        assert bins.ndim == 1
        # Bin indices stay within [0, n_bins).
        assert bins.min() >= 0
        assert bins.max() < 2_500

    def test_refractory_enforced(self):
        """All inter-spike intervals respect the absolute refractory.

        Drives a *very* high rate so the refractory is exercised on
        nearly every bin; the minimum observed gap must be at least
        ``refractory_bins + 1`` bins (the implementation contract,
        equivalent to ``> refractory_s`` for typical bin widths).
        """
        rng = np.random.default_rng(1)
        dt = 1e-4  # 0.1 ms bins
        refractory_s = 0.002  # 2 ms
        bins = poisson_train(
            rng,
            rate_profile=2_000.0,  # absurdly high rate to stress the limit
            dt=dt,
            n_bins=10_000,
            refractory_s=refractory_s,
        )
        assert bins.size > 100, "expected many spikes at this rate"
        gaps_bins = np.diff(bins)
        min_gap_bins = int(gaps_bins.min())
        refractory_bins = max(1, int(refractory_s / dt))
        # Implementation enforces `(b - last) > refractory_bins`, so the
        # smallest legal gap is `refractory_bins + 1` bins.
        assert min_gap_bins >= refractory_bins + 1, (
            f"minimum ISI {min_gap_bins * dt * 1e3:.4f} ms violates the "
            f"{refractory_s * 1e3:.1f} ms absolute refractory"
        )

    def test_zero_rate_produces_no_spikes(self):
        rng = np.random.default_rng(2)
        bins = poisson_train(rng, rate_profile=0.0, dt=1e-3, n_bins=1_000)
        assert bins.size == 0

    def test_reproducible_with_same_seed(self):
        bins1 = poisson_train(
            np.random.default_rng(7),
            rate_profile=20.0,
            dt=1e-3,
            n_bins=2_500,
        )
        bins2 = poisson_train(
            np.random.default_rng(7),
            rate_profile=20.0,
            dt=1e-3,
            n_bins=2_500,
        )
        np.testing.assert_array_equal(bins1, bins2)

    def test_accepts_array_rate_profile(self):
        """A length-``n_bins`` rate array is honoured pointwise."""
        rng = np.random.default_rng(3)
        rate = np.zeros(1_000)
        rate[500:] = 50.0  # rate is 0 before bin 500, 50 Hz after
        bins = poisson_train(rng, rate_profile=rate, dt=1e-3, n_bins=1_000)
        # All spikes must land in the high-rate half of the window.
        assert (bins >= 500).all(), "spikes leaked into the zero-rate window"


# ======================================================================
# make_tuned_spikes
# ======================================================================


class TestMakeTunedSpikes:
    def test_default_seed_reproducibility(self):
        """Calling with no seed twice yields bit-identical streams —
        this is the contract conftest.py relies on for its
        ``tuned_neuron`` fixture."""
        st1, tr1, ang1, uniq1 = make_tuned_spikes()
        st2, tr2, ang2, uniq2 = make_tuned_spikes()
        np.testing.assert_array_equal(st1, st2)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(ang1, ang2)
        np.testing.assert_array_equal(uniq1, uniq2)

    def test_shapes_and_dtypes(self):
        st, tr, angles, unique_angles = make_tuned_spikes(
            n_angles=8,
            n_repeats=10,
        )
        n_trials = 8 * 10
        assert angles.shape == (n_trials,)
        assert unique_angles.shape == (8,)
        assert st.shape == tr.shape
        assert tr.dtype == np.int64
        assert tr.max() < n_trials
        assert tr.min() >= 0
        # Spike times are trial-relative seconds.
        assert (st >= 0).all()
        assert (st <= 2.5).all()

    def test_tuning_concentrates_around_preferred(self):
        """Per-trial mean spike count is largest near the preferred
        angle.  We don't compute OSI here (the helper doesn't enforce
        a refractory, so single-spike skew is real); a simple peak
        check is enough."""
        st, tr, angles, unique_angles = make_tuned_spikes(
            preferred_angle=90.0,
            sigma_deg=15.0,
            n_repeats=30,
            peak_rate=40.0,
            base_rate=1.0,
            rng=11,
        )
        # Mean per-trial count per unique angle.
        counts = np.array([np.sum(tr == t) for t in range(len(angles))], dtype=np.float64)
        means_per_angle = np.array([counts[angles == ang].mean() for ang in unique_angles])
        pref_idx = int(np.argmin(np.abs(unique_angles - 90.0)))
        assert int(np.argmax(means_per_angle)) == pref_idx


# ======================================================================
# make_two_unit_demo
# ======================================================================


class TestMakeTwoUnitDemo:
    def test_returns_namedtuple_with_documented_fields(self):
        demo = make_two_unit_demo(seed=42)
        assert isinstance(demo, TwoUnitDemo)
        # Field set matches the docstring.
        expected = {
            "spike_times_c1",
            "trials_c1",
            "spike_times_c2",
            "trials_c2",
            "spike_times",
            "trials",
            "waveforms",
            "ground_truth",
            "angles",
            "n_trials",
            "n_orientations",
            "n_repeats",
            "c2_pref_ori",
            "c2_sigma",
            "waveform_fs",
            "sorting_data",
            "rng",
        }
        assert set(demo._fields) == expected

    def test_array_shapes_consistent(self):
        demo = make_two_unit_demo(seed=42, n_orientations=12, n_repeats=20)
        n_total = len(demo.spike_times_c1) + len(demo.spike_times_c2)
        assert demo.spike_times.shape == (n_total,)
        assert demo.trials.shape == (n_total,)
        assert demo.waveforms.shape == (n_total, 48)  # default snippet_len
        assert demo.ground_truth.shape == (n_total,)
        assert demo.angles.shape == (demo.n_trials,)
        assert demo.n_trials == 12 * 20

    def test_merged_spike_times_are_sorted(self):
        demo = make_two_unit_demo(seed=42)
        # ``argsort`` was used during merge — verify the post-condition.
        assert (np.diff(demo.spike_times) >= 0).all()

    def test_ground_truth_aligned_with_clusters(self):
        """Spikes flagged as ground-truth 0 must come from the C1
        train, and ground-truth 1 from C2.  Cardinalities must match
        the originating spike arrays exactly."""
        demo = make_two_unit_demo(seed=42)
        n_gt0 = int((demo.ground_truth == 0).sum())
        n_gt1 = int((demo.ground_truth == 1).sum())
        assert n_gt0 == len(demo.spike_times_c1)
        assert n_gt1 == len(demo.spike_times_c2)

    def test_c1_is_orientation_indifferent(self):
        """Cluster 1 fires at the same stimulus rate regardless of
        angle, so per-angle spike counts should be uniform within
        Poisson fluctuations.  We assert CV < 0.20 — comfortably
        above the expected ~7 % CV at the default rates, but well
        below anything a tuned neuron would produce."""
        demo = make_two_unit_demo(seed=42)
        # Per-orientation total C1 spike count.
        counts_per_ori = np.zeros(demo.n_orientations, dtype=np.float64)
        unique_oris = np.unique(demo.angles)
        for i, ori in enumerate(unique_oris):
            trial_idx_at_ori = np.where(demo.angles == ori)[0]
            mask = np.isin(demo.trials_c1, trial_idx_at_ori)
            counts_per_ori[i] = int(mask.sum())
        cv = counts_per_ori.std() / counts_per_ori.mean()
        assert cv < 0.20, (
            f"C1 was meant to be orientation-indifferent but has CV={cv:.3f} "
            f"across orientations: {counts_per_ori}"
        )

    def test_c2_is_orientation_selective(self):
        """Cluster 2 has a Gaussian tuning curve peaked at
        ``c2_pref_ori`` with width ``c2_sigma=15°``.  At the default
        parameters the gOSI should be effectively saturated (> 0.5)."""
        demo = make_two_unit_demo(seed=42)
        # Per-trial firing rate for cluster 2, then mean per angle.
        unique_oris = np.unique(demo.angles)
        rate_per_ori = np.zeros(len(unique_oris), dtype=np.float64)
        for i, ori in enumerate(unique_oris):
            trial_idx_at_ori = np.where(demo.angles == ori)[0]
            mask = np.isin(demo.trials_c2, trial_idx_at_ori)
            n_spikes = int(mask.sum())
            rate_per_ori[i] = n_spikes / len(trial_idx_at_ori)
        c2_gosi = gosi(rate_per_ori, unique_oris)
        assert c2_gosi > 0.5, (
            f"C2 was meant to be sharply tuned but gOSI={c2_gosi:.3f} "
            f"(rates per orientation: {rate_per_ori})"
        )

    def test_seeded_reproducibility(self):
        """Same seed → identical merged arrays."""
        d1 = make_two_unit_demo(seed=42)
        d2 = make_two_unit_demo(seed=42)
        np.testing.assert_array_equal(d1.spike_times, d2.spike_times)
        np.testing.assert_array_equal(d1.trials, d2.trials)
        np.testing.assert_array_equal(d1.ground_truth, d2.ground_truth)
        np.testing.assert_allclose(d1.waveforms, d2.waveforms)


# ======================================================================
# Public-API top-level re-export contract
# ======================================================================


class TestTopLevelReexports:
    """Regression test for the top-level barrel.

    The names below were already exposed in ``neural_cca.sorting.__all__``
    but missing from ``neural_cca.__all__`` — making
    ``from neural_cca import rpvs`` fail even though it works through
    the subpackage. Make sure the barrel keeps them.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "neg_silhouette_score",
            "spikes_before_stimulus",
            "est_snr",
            "calc_weighted_snr",
            "rpvs",
        ],
    )
    def test_importable_from_top_level(self, name):
        import neural_cca as nc

        assert hasattr(nc, name), (
            f"neural_cca.{name} should be importable from the top-level barrel"
        )
        assert callable(getattr(nc, name))
        assert name in nc.__all__, f"{name} should be listed in neural_cca.__all__"
