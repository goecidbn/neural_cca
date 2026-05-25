"""Biologically inspired tests for orientation selectivity functions.

Each test creates synthetic neural data with known properties
(simple cell, complex cell, sharply tuned, etc.) and verifies that
the analysis functions recover the expected characteristics.

Run with:
    python -m pytest tests/test_tuning.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca._utils import guarded_divide, steps2degree
from neural_cca.sta.analysis import calc_mfr_trial
from neural_cca.tuning.selectivity import circular_variance, dosi_circular_normalised
from neural_cca.tuning.tuning import (
    compute_f0_f1_f2,
    get_os_metrics,
    preferred_dori,
    tuning_bandwidth,
)
from tests.conftest import make_psth as _make_psth_helper

# Shared helpers from conftest.py
from tests.conftest import make_tuned_spikes as _make_tuned_spikes

# ======================================================================
# Tests: Selectivity indices
# ======================================================================


class TestOSI:
    """Tests for orientation selectivity index."""

    def test_sharply_tuned_neuron_high_osi(self):
        """A sharply tuned neuron should have OSI > 0.3.

        OSI doubles angles (orientation space), so responses must be
        symmetric about 180° — strong at 0° *and* 180°, weak at 90°
        and 270° — to appear sharply tuned.
        """
        # 8 angles: 0, 45, 90, 135, 180, 225, 270, 315
        # Strong at 0° & 180° (preferred orientation), weak at 90° & 270°
        activities = np.array([20, 5, 1, 5, 20, 5, 1, 5], dtype=float)
        osi = dosi_circular_normalised(activities, 8)
        assert osi > 0.3, f"Sharply tuned: expected OSI > 0.3, got {osi:.3f}"

    def test_untuned_neuron_zero_osi(self):
        """An untuned neuron should have OSI near 0."""
        activities = np.ones(12) * 10.0
        osi = dosi_circular_normalised(activities, 12)
        assert osi < 0.01, f"Untuned: expected OSI ≈ 0, got {osi:.3f}"

    def test_perfectly_tuned_neuron(self):
        """Response at only one orientation should give OSI = 1."""
        activities = np.zeros(8)
        activities[0] = 10.0
        osi = dosi_circular_normalised(activities, 8)
        assert osi == pytest.approx(1.0, abs=0.001)

    def test_direction_selectivity_asymmetric(self):
        """Asymmetric responses (preferred vs anti-preferred) → DSI > 0.3."""
        # Strong at 0°, weak at 180°
        angles = np.linspace(0, 360, 8, endpoint=False)
        activities = np.array([20, 10, 3, 2, 1, 2, 3, 10], dtype=float)
        dsi = dosi_circular_normalised(activities, angles, direction_selectivity=True)
        assert dsi > 0.3, f"Direction selective: expected DSI > 0.3, got {dsi:.3f}"

    def test_direction_non_selective_symmetric(self):
        """Symmetric response around 0° and 180° → DSI near 0."""
        # Equal response at 0° and 180° (orientation-selective but not
        # direction-selective)
        activities = np.array([20, 5, 2, 5, 20, 5, 2, 5], dtype=float)
        dsi = dosi_circular_normalised(activities, 8, direction_selectivity=True)
        assert dsi < 0.1, f"Non-dir-selective: expected DSI < 0.1, got {dsi:.3f}"

    def test_explicit_angles_same_as_int(self):
        """Explicit angle array should match int shorthand."""
        activities = np.array([10, 8, 5, 2, 1, 2, 5, 8], dtype=float)
        osi_int = dosi_circular_normalised(activities, 8)
        angles = np.linspace(0, 360, 8, endpoint=False)
        osi_arr = dosi_circular_normalised(activities, angles)
        assert osi_int == pytest.approx(osi_arr, abs=1e-10)


class TestCircularVariance:
    """Tests for circular variance."""

    def test_sharp_tuning_low_cirvar(self):
        """Sharp tuning → CirVar < 0.7 (orientation-symmetric profile)."""
        # Same orientation-symmetric profile as OSI test
        activities = np.array([20, 5, 1, 5, 20, 5, 1, 5], dtype=float)
        angles = np.linspace(0, 360, 8, endpoint=False)
        cvar = circular_variance(activities, angles)
        assert cvar < 0.7, f"Sharp tuning: expected CirVar < 0.7, got {cvar:.3f}"

    def test_flat_response_high_cirvar(self):
        """Flat response → CirVar ≈ 1."""
        activities = np.ones(12) * 5.0
        angles = np.linspace(0, 360, 12, endpoint=False)
        cvar = circular_variance(activities, angles)
        assert cvar > 0.99, f"Flat: expected CirVar ≈ 1, got {cvar:.3f}"

    def test_cirvar_equals_one_minus_osi(self):
        """CirVar should equal 1 - OSI by definition."""
        activities = np.array([10, 8, 3, 1, 3, 8], dtype=float)
        angles = np.linspace(0, 360, 6, endpoint=False)
        osi = dosi_circular_normalised(activities, angles)
        cvar = circular_variance(activities, angles)
        assert cvar == pytest.approx(1 - osi, abs=1e-10)


# ======================================================================
# Tests: Preferred orientation
# ======================================================================


class TestPreferredOrientation:
    """Tests for preferred_dori."""

    def test_peak_at_90_degrees(self):
        """Neuron with peak response at 90° → preferred near 90°."""
        angles = np.linspace(0, 360, 12, endpoint=False)
        # Create Gaussian response centred at 90°
        activities = 2.0 + 10.0 * np.exp(-((angles - 90) ** 2) / (2 * 30**2))
        pref = preferred_dori(activities, angles)
        assert 80 < pref < 100, f"Expected ~90°, got {pref:.1f}°"

    def test_peak_at_0_degrees(self):
        """Neuron responding at 0° and 180° (orientation space)."""
        # Strong response at 0° and 180° (since OSI doubles angles)
        angles = np.linspace(0, 360, 8, endpoint=False)
        activities = np.array([20, 5, 2, 5, 20, 5, 2, 5], dtype=float)
        pref = preferred_dori(activities, angles)
        assert pref < 10 or pref > 170, f"Expected ~0° or ~180°, got {pref:.1f}°"

    def test_direction_preferred_at_45(self):
        """Direction-selective neuron preferring 45°."""
        angles = np.linspace(0, 360, 12, endpoint=False)
        activities = 2.0 + 15.0 * np.exp(-((angles - 45) ** 2) / (2 * 25**2))
        pref_dir = preferred_dori(activities, angles, direction_selectivity=True)
        assert 30 < pref_dir < 60, f"Expected ~45°, got {pref_dir:.1f}°"


# ======================================================================
# Tests: Tuning bandwidth
# ======================================================================


class TestTuningBandwidth:
    """Tests for tuning_bandwidth (HWHH of Gaussian fit)."""

    def test_narrow_tuning(self):
        """Gaussian with sigma=20° → HWHH ≈ 23.5°."""
        orientations = np.linspace(0, 170, 18)
        sigma = 20.0
        expected_hwhh = sigma * np.sqrt(2 * np.log(2))  # ≈ 23.5°
        responses = 2.0 + 10.0 * np.exp(-((orientations - 90) ** 2) / (2 * sigma**2))
        bw = tuning_bandwidth(responses, orientations)
        assert abs(bw - expected_hwhh) < 5, f"Expected HWHH ≈ {expected_hwhh:.1f}°, got {bw:.1f}°"

    def test_broad_tuning(self):
        """Gaussian with sigma=60° → broader bandwidth."""
        orientations = np.linspace(0, 170, 18)
        sigma = 60.0
        responses = 2.0 + 10.0 * np.exp(-((orientations - 90) ** 2) / (2 * sigma**2))
        bw = tuning_bandwidth(responses, orientations)
        assert bw > 40, f"Expected broad bandwidth > 40°, got {bw:.1f}°"

    def test_flat_response_returns_inf(self):
        """Flat responses should return np.inf."""
        orientations = np.linspace(0, 170, 18)
        responses = np.ones_like(orientations) * 5.0
        bw = tuning_bandwidth(responses, orientations)
        assert np.isinf(bw), f"Expected inf for flat response, got {bw}"


# ======================================================================
# Tests: F0 / F1 / F2 harmonics
# ======================================================================


class TestF0F1F2:
    """Tests for compute_f0_f1_f2 — simple vs complex cell classification."""

    @staticmethod
    def _make_psth(
        f_stim: float,
        duration: float,
        bin_size: float,
        dc: float,
        f1_amp: float,
        f2_amp: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """Delegate to conftest.make_psth."""
        return _make_psth_helper(f_stim, duration, bin_size, dc, f1_amp, f2_amp)

    def test_simple_cell_high_f1_f0(self):
        """Simple cell: strong modulation at f_stim → F1/F0 > 1."""
        psth, fs = self._make_psth(
            f_stim=2.0,
            duration=2.0,
            bin_size=0.01,
            dc=5.0,
            f1_amp=8.0,
        )
        F0, F1, F2 = compute_f0_f1_f2(psth, fs, f_stim=2.0)
        ratio = float(F1 / F0)
        assert ratio > 1.0, f"Simple cell: expected F1/F0 > 1, got {ratio:.3f}"

    def test_complex_cell_low_f1_f0(self):
        """Complex cell: weak modulation → F1/F0 < 1."""
        psth, fs = self._make_psth(
            f_stim=2.0,
            duration=2.0,
            bin_size=0.01,
            dc=10.0,
            f1_amp=1.0,
        )
        F0, F1, F2 = compute_f0_f1_f2(psth, fs, f_stim=2.0)
        ratio = float(F1 / F0)
        assert ratio < 1.0, f"Complex cell: expected F1/F0 < 1, got {ratio:.3f}"

    def test_f2_dominance(self):
        """Frequency-doubled response → F2 > F1."""
        psth, fs = self._make_psth(
            f_stim=2.0,
            duration=2.0,
            bin_size=0.01,
            dc=5.0,
            f1_amp=1.0,
            f2_amp=6.0,
        )
        F0, F1, F2 = compute_f0_f1_f2(psth, fs, f_stim=2.0)
        assert float(F2) > float(F1), (
            f"Expected F2 > F1, got F1={float(F1):.3f}, F2={float(F2):.3f}"
        )

    def test_dc_recovery(self):
        """F0 should recover the mean firing rate."""
        dc = 7.5
        psth, fs = self._make_psth(
            f_stim=2.0,
            duration=2.0,
            bin_size=0.01,
            dc=dc,
            f1_amp=3.0,
        )
        F0, _, _ = compute_f0_f1_f2(psth, fs, f_stim=2.0)
        # With rectification, F0 will be slightly above dc
        assert float(F0) >= dc * 0.95, f"F0 should approximate DC={dc}, got {float(F0):.3f}"

    def test_pure_sinusoid_amplitude(self):
        """Pure sinusoid: F1 amplitude should match input amplitude."""
        # No rectification needed (dc > amplitude)
        dc, amp = 10.0, 3.0
        psth, fs = self._make_psth(
            f_stim=2.0,
            duration=2.0,
            bin_size=0.01,
            dc=dc,
            f1_amp=amp,
        )
        F0, F1, _ = compute_f0_f1_f2(psth, fs, f_stim=2.0)
        assert float(F1) == pytest.approx(amp, abs=0.3), f"Expected F1 ≈ {amp}, got {float(F1):.3f}"

    def test_multi_trial_batch(self):
        """F0/F1/F2 should work on batched PSTHs (n_trials, time)."""
        bin_size = 0.01
        t = np.arange(0, 2.0, bin_size)
        n_trials = 10
        psth = np.zeros((n_trials, len(t)))
        for i in range(n_trials):
            psth[i] = 5.0 + 3.0 * np.sin(2 * np.pi * 2.0 * t + i * 0.5)

        F0, F1, F2 = compute_f0_f1_f2(psth, 1.0 / bin_size, f_stim=2.0)
        assert F0.shape == (n_trials,)
        assert F1.shape == (n_trials,)
        assert all(F1 > 2.0), "All trials should have F1 > 2 for amp=3"


# ======================================================================
# Tests: Trial-wise firing rates
# ======================================================================


class TestCalcMfrTrial:
    """Tests for calc_mfr_trial."""

    def test_known_rate(self):
        """10 spikes in a 2s stimulus → MFR = 5 Hz."""
        stim_onset = 0.5
        trial_dur = 2.5
        stim_dur = trial_dur - stim_onset  # 2.0 s
        # 10 evoked spikes in trial 0
        spike_times = np.concatenate(
            [
                np.array([0.1, 0.2]),  # spontaneous
                np.linspace(0.6, 2.4, 10),  # evoked
            ]
        )
        trials = np.zeros(12, dtype=np.int64)

        mfr = calc_mfr_trial(
            spike_times,
            trials,
            stim_window=(stim_onset, trial_dur),
            n_trials=1,
        )
        assert mfr[0] == pytest.approx(10.0 / stim_dur, abs=0.01)

    def test_cluster_filtering(self):
        """Only count spikes from the requested cluster."""
        spike_times = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        trials = np.zeros(5, dtype=np.int64)
        cluster_labels = np.array([0, 0, 1, 1, 1])

        mfr = calc_mfr_trial(
            spike_times,
            trials,
            all_clusters=False,
            cluster_labels=cluster_labels,
            cluster_id=0,
            stim_window=(0.5, 2.5),
            n_trials=1,
        )
        # cluster 0 has 2 spikes, stim_duration = 2.0
        assert mfr[0] == pytest.approx(2.0 / 2.0, abs=0.01)


# ======================================================================
# Tests: Full get_os_metrics pipeline
# ======================================================================


class TestGetOsMetrics:
    """Integration tests for get_os_metrics using synthetic neurons."""

    def test_tuned_neuron_metrics(self):
        """Sharply tuned neuron should produce high OSI, low CirVar."""
        st, tr, angles, angles_deg = _make_tuned_spikes(
            preferred_angle=90.0,
            sigma_deg=25.0,
            peak_rate=30.0,
            base_rate=2.0,
        )
        metrics = get_os_metrics(
            st,
            tr,
            angles,
            stim_window=(0.5, 2.5),
            stim_frequency=2.0,
            return_verbose=1,
        )
        assert metrics["osi"] > 0.2, f"Tuned: osi={metrics['osi']:.3f}"
        assert metrics["circular_variance"] < 0.85, (
            f"Tuned: circular_variance={metrics['circular_variance']:.3f}"
        )
        # Preferred orientation should be near 90°
        assert 60 < metrics["preferred_orientation"] < 120, (
            f"Expected pref ≈ 90°, got {metrics['preferred_orientation']:.1f}°"
        )

    def test_untuned_neuron_metrics(self):
        """Untuned neuron → OSI near 0."""
        # All angles get same rate
        st, tr, angles, _ = _make_tuned_spikes(
            preferred_angle=0.0,
            sigma_deg=9999.0,
            peak_rate=10.0,
            base_rate=10.0,
        )
        metrics = get_os_metrics(
            st,
            tr,
            angles,
            stim_window=(0.5, 2.5),
            stim_frequency=2.0,
            return_verbose=0,
        )
        assert metrics["osi"] < 0.15, f"Untuned: osi={metrics['osi']:.3f}"

    def test_verbose_levels(self):
        """return_verbose=2 should include intermediate arrays."""
        st, tr, angles, _ = _make_tuned_spikes()
        m0 = get_os_metrics(st, tr, angles, return_verbose=0)
        m1 = get_os_metrics(st, tr, angles, return_verbose=1)
        m2 = get_os_metrics(st, tr, angles, return_verbose=2)

        assert "f0_mean" not in m0
        assert "f0_mean" in m1
        assert "mfrs" in m2
        assert "psth_by_trial" in m2


# ======================================================================
# Tests: Utility functions
# ======================================================================


class TestUtils:
    """Tests for guarded_divide and steps2degree."""

    def test_guarded_divide_zero_denom(self):
        """Zero denominator → return numerator."""
        result = guarded_divide(np.array([3.0, 5.0]), np.array([0.0, 2.0]))
        np.testing.assert_array_almost_equal(result, [3.0, 2.5])

    def test_guarded_divide_scalar(self):
        """Scalar inputs should return float."""
        assert guarded_divide(4.0, 0.0) == 4.0
        assert guarded_divide(4.0, 2.0) == pytest.approx(2.0)

    def test_steps2degree_12(self):
        """12 steps → 30° spacing starting at 0°."""
        d = steps2degree(12)
        assert len(d) == 12
        assert d[1] == 0.0
        assert d[2] == pytest.approx(30.0)
        assert d[12] == pytest.approx(330.0)

    def test_steps2degree_4(self):
        """4 steps → 90° spacing."""
        d = steps2degree(4)
        assert d == {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0}


# ======================================================================
# Tests: Extended get_os_metrics (v0.3.0)
# ======================================================================


class TestGetOsMetricsExtended:
    """Tests for new v0.3.0 params on get_os_metrics."""

    def test_compute_gosi(self):
        """compute_gosi=True adds gosi and gdsi keys."""
        st, tr, angles, _ = _make_tuned_spikes(preferred_angle=90.0, sigma_deg=25.0)
        m = get_os_metrics(st, tr, angles, compute_gosi=True, return_verbose=0)
        assert "gosi" in m
        assert "gdsi" in m
        assert m["gosi"] > 0.1  # tuned neuron

    def test_compute_p_values(self):
        """compute_p_values=True adds p-value keys."""
        st, tr, angles, _ = _make_tuned_spikes(preferred_angle=90.0, sigma_deg=25.0)
        m = get_os_metrics(
            st,
            tr,
            angles,
            compute_p_values=True,
            compute_gosi=True,
            return_verbose=0,
        )
        assert "osi_p_value" in m
        assert "dsi_p_value" in m
        assert "gosi_p_value" in m
        assert "anova_p_value" in m

    def test_fit_model_von_mises(self):
        """fit_model='von_mises_orientation' adds fitting keys."""
        st, tr, angles, _ = _make_tuned_spikes(preferred_angle=90.0, sigma_deg=25.0)
        m = get_os_metrics(
            st,
            tr,
            angles,
            fit_model="von_mises_orientation",
            return_verbose=0,
        )
        assert m["fit_model"] == "von_mises_orientation"
        assert "fit_r_squared" in m
        assert "fit_preferred_angle" in m

    def test_bootstrap_ci(self):
        """bootstrap_ci=True adds CI dicts."""
        st, tr, angles, _ = _make_tuned_spikes(preferred_angle=90.0, sigma_deg=25.0)
        m = get_os_metrics(
            st,
            tr,
            angles,
            bootstrap_ci=True,
            n_bootstrap=50,
            compute_gosi=True,
            return_verbose=0,
        )
        assert "osi_ci" in m
        assert "ci_lower" in m["osi_ci"]
        assert "ci_upper" in m["osi_ci"]
        assert "gosi_ci" in m


# ======================================================================
# Tests: per-trial spike filter is built exactly once per get_os_metrics
# ======================================================================


class TestTrialFilterReuse:
    """Regression tests for the _TrialFilteredSpikes refactor.

    Before the refactor, ``get_os_metrics(compute_p_values=True, ...)``
    walked the spike arrays multiple times — once at the top of the
    function and once inside ``anova_across_orientations`` — because
    each consumer rebuilt the per-trial filter from raw arrays.  These
    tests pin the new contract: the filter is built exactly once per
    call and shared with every downstream helper that needs it.
    """

    def test_build_trial_filter_called_exactly_once(self):
        """Even with ANOVA, gOSI, and bootstrap CI all enabled."""
        from unittest.mock import patch

        from neural_cca.tuning import (
            _filter as _filter_mod,
        )
        from neural_cca.tuning import (
            tuning as _tuning_mod,
        )

        st, tr, angles, _ = _make_tuned_spikes(
            preferred_angle=90.0,
            sigma_deg=25.0,
        )

        # Patch the name as imported into ``tuning`` (not the
        # definition in ``_filter``) so the wrapper sees calls made
        # from ``get_os_metrics``.  ``wraps=`` keeps the real
        # implementation in place so the metrics still compute.
        with patch.object(
            _tuning_mod,
            "_build_trial_filter",
            wraps=_filter_mod._build_trial_filter,
        ) as mock_build:
            get_os_metrics(
                st,
                tr,
                angles,
                compute_p_values=True,
                compute_gosi=True,
                bootstrap_ci=True,
                n_bootstrap=20,
                return_verbose=1,
            )

        assert mock_build.call_count == 1, (
            f"_build_trial_filter was called {mock_build.call_count} "
            f"times; expected exactly 1.  A consumer of the filter "
            f"is rebuilding it instead of accepting the pre-built one."
        )

    def test_anova_skips_rebuild_when_filter_passed(self):
        """anova_across_orientations does not rebuild when given _filter."""
        from unittest.mock import patch

        from neural_cca.tuning import (
            _filter as _filter_mod,
        )
        from neural_cca.tuning import (
            statistics as _stats_mod,
        )
        from neural_cca.tuning._filter import (
            _build_trial_filter,
        )
        from neural_cca.tuning.statistics import (
            anova_across_orientations,
        )

        st, tr, angles, _ = _make_tuned_spikes(
            preferred_angle=90.0,
            sigma_deg=25.0,
        )
        prebuilt = _build_trial_filter(
            st,
            tr,
            angles,
            stim_window=(0.5, 2.5),
        )

        with patch.object(
            _stats_mod,
            "_build_trial_filter",
            wraps=_filter_mod._build_trial_filter,
        ) as mock_build:
            result = anova_across_orientations(
                st,
                tr,
                angles,
                stim_window=(0.5, 2.5),
                _filter=prebuilt,
            )

        assert mock_build.call_count == 0
        # Sanity: the function still produces a usable result.
        assert "p_value" in result
        assert not np.isnan(result["p_value"])

    def test_anova_rebuilds_when_no_filter_passed(self):
        """Public callers without a filter still get a fresh build."""
        from unittest.mock import patch

        from neural_cca.tuning import (
            _filter as _filter_mod,
        )
        from neural_cca.tuning import (
            statistics as _stats_mod,
        )
        from neural_cca.tuning.statistics import (
            anova_across_orientations,
        )

        st, tr, angles, _ = _make_tuned_spikes(
            preferred_angle=90.0,
            sigma_deg=25.0,
        )
        with patch.object(
            _stats_mod,
            "_build_trial_filter",
            wraps=_filter_mod._build_trial_filter,
        ) as mock_build:
            anova_across_orientations(
                st,
                tr,
                angles,
                stim_window=(0.5, 2.5),
            )
        assert mock_build.call_count == 1
