"""Tests for tuning curve fitting functions.

Run with:
    python -m pytest tests/test_tuning_fitting.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_cca.tuning.fitting import (
    double_gaussian_fit,
    goodness_of_fit,
    tuning_curve_interpolation,
    von_mises_fit,
)
from tests.conftest import (
    make_double_gaussian_response as _double_gaussian_response,
)

# Shared helpers from conftest.py
from tests.conftest import (
    make_von_mises_response as _von_mises_response,
)

# ======================================================================
# Tests: von Mises fit
# ======================================================================


class TestVonMisesFit:
    def test_recovers_known_params(self):
        """Fit should recover kappa and preferred orientation."""
        oris = np.linspace(0, 170, 18)
        resp = _von_mises_response(oris, kappa=3.0, theta0_deg=90.0)
        result = von_mises_fit(resp, oris, tuning_type="orientation")
        assert result["tuning_type"] == "orientation"
        assert result["r_squared"] > 0.95
        assert abs(result["preferred_angle"] - 90.0) < 15.0
        assert result["kappa"] > 1.0

    def test_return_fit_gives_tuple(self):
        """return_fit=True should return (dict, array)."""
        oris = np.linspace(0, 170, 18)
        resp = _von_mises_response(oris)
        result, fitted = von_mises_fit(
            resp,
            oris,
            tuning_type="orientation",
            return_fit=True,
        )
        assert isinstance(result, dict)
        assert len(fitted) == len(oris)

    def test_bandwidth_is_finite(self):
        """Well-tuned curve should give finite bandwidth."""
        oris = np.linspace(0, 170, 18)
        resp = _von_mises_response(oris, kappa=3.0)
        result = von_mises_fit(resp, oris, tuning_type="orientation")
        assert np.isfinite(result["bandwidth_hwhh"])
        assert result["bandwidth_hwhh"] > 0

    def test_flat_response_nan(self):
        """Flat response should produce NaN results."""
        oris = np.linspace(0, 170, 18)
        resp = np.ones(18) * 5.0
        result = von_mises_fit(resp, oris, tuning_type="orientation")
        # Fit may succeed with near-zero amplitude or fail → NaN
        # Either is acceptable
        assert np.isnan(result["r_squared"]) or result["r_squared"] < 0.1

    def test_direction_mode_recovers_two_bumps(self):
        """Direction-mode fit on a sum-of-two-bumps response."""
        dirs = np.linspace(0, 350, 36)
        resp = _von_mises_response(dirs, kappa=2.0, theta0_deg=60.0)
        result = von_mises_fit(resp, dirs, tuning_type="direction")
        assert result["tuning_type"] == "direction"
        assert result["r_squared"] > 0.7
        assert "amplitude_pref" in result
        assert "amplitude_null" in result
        assert "ds_ratio" in result

    def test_invalid_tuning_type_raises(self):
        """Unknown tuning_type should raise ValueError."""
        oris = np.linspace(0, 170, 18)
        resp = _von_mises_response(oris)
        with pytest.raises(ValueError, match="tuning_type"):
            von_mises_fit(resp, oris, tuning_type="bogus")  # type: ignore[arg-type]


# ======================================================================
# Tests: double Gaussian fit
# ======================================================================


class TestDoubleGaussianFit:
    def test_recovers_bimodal(self):
        """Should capture bimodal tuning curve."""
        oris = np.linspace(0, 350, 36)
        resp = _double_gaussian_response(oris, A1=10.0, A2=5.0, theta0_deg=45.0)
        result = double_gaussian_fit(resp, oris)
        assert result["r_squared"] > 0.8

    def test_return_fit_tuple(self):
        """return_fit=True returns tuple."""
        oris = np.linspace(0, 350, 36)
        resp = _double_gaussian_response(oris)
        result, fitted = double_gaussian_fit(resp, oris, return_fit=True)
        assert isinstance(result, dict)
        assert len(fitted) == len(oris)


# ======================================================================
# Tests: tuning curve interpolation
# ======================================================================


class TestTuningCurveInterpolation:
    def test_preferred_near_true(self):
        """Interpolated preferred orientation within ±10° of true."""
        oris = np.linspace(0, 170, 12)
        resp = _von_mises_response(oris, kappa=3.0, theta0_deg=73.0)
        pref = tuning_curve_interpolation(
            resp,
            oris,
            model="von_mises_orientation",
        )
        assert abs(pref - 73.0) < 10.0, f"Expected ~73°, got {pref:.1f}°"

    def test_invalid_model_raises(self):
        """Unknown model string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            tuning_curve_interpolation(np.ones(8), np.arange(8) * 22.5, model="invalid")


# ======================================================================
# Tests: goodness of fit
# ======================================================================


class TestGoodnessOfFit:
    def test_perfect_fit(self):
        """Identical observed and predicted → R² = 1."""
        obs = np.array([1, 2, 3, 4, 5], dtype=float)
        assert goodness_of_fit(obs, obs) == pytest.approx(1.0)

    def test_zero_variance(self):
        """Constant observed → NaN."""
        obs = np.ones(5) * 3.0
        assert np.isnan(goodness_of_fit(obs, np.ones(5) * 2.0))

    def test_poor_fit(self):
        """Random prediction on structured data → low R²."""
        rng = np.random.default_rng(42)
        obs = np.arange(10, dtype=float)
        pred = rng.uniform(0, 10, 10)
        r2 = goodness_of_fit(obs, pred)
        assert r2 < 0.5
