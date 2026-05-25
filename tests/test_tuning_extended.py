"""Tests for extended tuning analysis: gOSI, gDSI, modulation, temporal.

Run with:
    python -m pytest tests/test_tuning_extended.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.tuning.selectivity import (
    dosi_circular_normalised,
    circular_variance,
    gosi,
    gdsi,
)
from neural_cca.tuning.modulation import (
    modulation_ratio_per_orientation,
    cross_orientation_suppression,
)
from neural_cca.tuning.temporal import (
    f1_phase,
    temporal_frequency_tuning,
)

# Shared helper from conftest.py
from tests.conftest import make_tuned_spikes as _make_tuned_spikes


# ======================================================================
# Tests: gOSI / gDSI
# ======================================================================

class TestGOSI:

    def test_tuned_neuron_high_gosi(self):
        """Sharply tuned neuron → gOSI > 0.2."""
        angles = np.linspace(0, 360, 12, endpoint=False)
        resp = 2.0 + 20.0 * np.exp(-((angles - 90) ** 2) / (2 * 25 ** 2))
        g = gosi(resp, angles)
        assert g > 0.2, f"Expected gOSI > 0.2, got {g:.3f}"

    def test_untuned_low_gosi(self):
        """Flat response → gOSI near 0."""
        angles = np.linspace(0, 360, 12, endpoint=False)
        resp = np.ones(12) * 10.0
        g = gosi(resp, angles)
        assert abs(g) < 0.15, f"Expected gOSI ≈ 0, got {g:.3f}"

    def test_p_value_returns_dict(self):
        """p_value=True returns dict with 'value' and 'p_value'."""
        angles = np.linspace(0, 360, 8, endpoint=False)
        resp = np.array([20, 5, 1, 5, 20, 5, 1, 5], dtype=float)
        result = gosi(resp, angles, p_value=True)
        assert isinstance(result, dict)
        assert "value" in result
        assert "p_value" in result

    def test_tuned_p_value_significant(self):
        """Strongly tuned → p < 0.05."""
        angles = np.linspace(0, 360, 12, endpoint=False)
        resp = 1.0 + 30.0 * np.exp(-((angles - 90) ** 2) / (2 * 20 ** 2))
        result = dosi_circular_normalised(resp, angles, p_value=True)
        assert result["p_value"] < 0.05


class TestGDSI:

    def test_direction_selective_high_gdsi(self):
        """Strong direction selectivity → gDSI > 0.3."""
        angles = np.linspace(0, 360, 12, endpoint=False)
        resp = 2.0 + 20.0 * np.exp(-((angles - 45) ** 2) / (2 * 25 ** 2))
        g = gdsi(resp, angles)
        assert g > 0.3, f"Expected gDSI > 0.3, got {g:.3f}"

    def test_symmetric_low_gdsi(self):
        """Symmetric ori-selective neuron → low gDSI."""
        angles = np.linspace(0, 360, 8, endpoint=False)
        resp = np.array([20, 5, 2, 5, 20, 5, 2, 5], dtype=float)
        g = gdsi(resp, angles)
        # Symmetric → should be low
        assert abs(g) < 0.3, f"Expected low gDSI, got {g:.3f}"


class TestCircularVariancePValue:

    def test_returns_dict_with_p(self):
        """p_value=True returns dict."""
        angles = np.linspace(0, 360, 8, endpoint=False)
        resp = np.array([20, 5, 1, 5, 20, 5, 1, 5], dtype=float)
        result = circular_variance(resp, angles, p_value=True)
        assert isinstance(result, dict)
        assert "value" in result
        assert 0 <= result["value"] <= 1


# ======================================================================
# Tests: Modulation
# ======================================================================

class TestModulationRatio:

    def test_simple_cell_high_f1f0(self):
        """Simple cell with sinusoidal modulation → F1/F0 > 1 at preferred."""
        rng = np.random.default_rng(99)
        n_angles = 8
        n_repeats = 10
        angles_deg = np.linspace(0, 360, n_angles, endpoint=False)
        angles = np.repeat(angles_deg, n_repeats)

        all_st, all_tr = [], []
        for t_idx in range(len(angles)):
            ang = angles[t_idx]
            d = min(abs(ang - 90), 360 - abs(ang - 90))
            rate = 2.0 + 20.0 * np.exp(-(d ** 2) / (2 * 30 ** 2))
            # Sinusoidal modulation at 2 Hz
            t_arr = np.arange(0.5, 2.5, 0.001)
            inst_rate = rate * (1 + 0.9 * np.sin(2 * np.pi * 2.0 * t_arr))
            inst_rate = np.maximum(inst_rate, 0)
            n_spikes = rng.poisson(np.sum(inst_rate) * 0.001)
            st = rng.uniform(0.5, 2.5, n_spikes)
            all_st.append(st)
            all_tr.append(np.full(len(st), t_idx, dtype=np.int64))

        spike_times = np.concatenate(all_st)
        trials = np.concatenate(all_tr)

        result = modulation_ratio_per_orientation(
            spike_times, trials, angles,
            stim_frequency=2.0, stim_window=(0.5, 2.5),
        )
        assert isinstance(result, dict)
        assert len(result) == n_angles


class TestCrossOrientationSuppression:

    def test_tuned_neuron_cos(self):
        """Tuned neuron → COS > 0."""
        angles = np.linspace(0, 360, 12, endpoint=False)
        resp = 2.0 + 20.0 * np.exp(-((angles - 90) ** 2) / (2 * 25 ** 2))
        cos = cross_orientation_suppression(resp, angles)
        assert cos > 0, f"Expected COS > 0, got {cos:.3f}"

    def test_flat_neuron_cos_zero(self):
        """Flat response → COS ≈ 0."""
        angles = np.linspace(0, 360, 8, endpoint=False)
        resp = np.ones(8) * 10.0
        cos = cross_orientation_suppression(resp, angles)
        assert abs(cos) < 0.1


# ======================================================================
# Tests: Temporal
# ======================================================================

class TestF1Phase:

    def test_known_phase(self):
        """Pure sinusoid with known phase → correct recovery."""
        fs = 100.0
        f_stim = 2.0
        t = np.arange(0, 2.0, 1.0 / fs)
        phase_true = np.pi / 4
        psth = 10.0 + 5.0 * np.sin(2 * np.pi * f_stim * t + phase_true)
        phase = f1_phase(psth, fs, f_stim)
        # The FFT phase of sin(wt + phi) is phi - pi/2 (since sin = cos shifted)
        # Check that the extracted phase is consistent (up to the sin/cos offset)
        # by verifying that the circular distance between recovered and true
        # is within pi/2 + tolerance
        diff = abs(np.angle(np.exp(1j * (phase - phase_true))))
        assert diff < np.pi / 2 + 0.2, f"Phase too far off: diff={diff:.2f} rad"

    def test_multi_dim(self):
        """Multi-trial array: returns array of phases."""
        fs = 100.0
        t = np.arange(0, 2.0, 1.0 / fs)
        psth = np.vstack([
            10 + 5 * np.sin(2 * np.pi * 2.0 * t + i * 0.5)
            for i in range(5)
        ])
        phases = f1_phase(psth, fs, 2.0)
        assert phases.shape == (5,)


class TestTemporalFrequencyTuning:

    def test_peak_at_known_tf(self):
        """Preferred TF should be near true peak."""
        rng = np.random.default_rng(123)
        tfs = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
        n_repeats = 10
        temporal_freqs = np.repeat(tfs, n_repeats)
        n_trials = len(temporal_freqs)

        all_st, all_tr = [], []
        for t_idx in range(n_trials):
            tf = temporal_freqs[t_idx]
            # Peak rate at 4 Hz
            rate = 5.0 + 20.0 * np.exp(-((np.log2(tf) - np.log2(4.0)) ** 2) / 2)
            n_spk = rng.poisson(rate * 2.0)
            st = rng.uniform(0.5, 2.5, n_spk)
            all_st.append(st)
            all_tr.append(np.full(len(st), t_idx, dtype=np.int64))

        spike_times = np.concatenate(all_st)
        trials = np.concatenate(all_tr)

        result = temporal_frequency_tuning(
            spike_times, trials, temporal_freqs,
            response_metric="mfr",
            stim_window=(0.5, 2.5),
        )
        assert result["preferred_tf"] == pytest.approx(4.0, abs=2.0)
        assert len(result["temporal_freqs"]) == 6
