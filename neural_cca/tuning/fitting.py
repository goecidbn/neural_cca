"""Tuning curve fitting models.

Provides von Mises (orientation or direction) and double Gaussian fits
for tuning curves, plus interpolation and goodness-of-fit utilities.

The single :func:`von_mises_fit` entry point handles both *orientation*
selectivity (single bump on the half-circle, ``cos(2*(θ-θ0))``) and
*direction* selectivity (two bumps at θ₀ and θ₀+π) via a
``tuning_type`` parameter.  Use ``tuning_type="orientation"`` for OS
data sampled on 0–180° or for direction data where you only care about
axial selectivity, and ``tuning_type="direction"`` when you need
separate preferred / null amplitudes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from .._utils import wrap180, wrap360

TuningType = Literal["orientation", "direction"]

__all__ = [
    "von_mises_fit",
    "double_gaussian_fit",
    "tuning_curve_interpolation",
    "goodness_of_fit",
]


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------


def _vm_orientation_model(
    theta: npt.NDArray,
    R0: float,
    A: float,
    kappa: float,
    theta0: float,
) -> np.ndarray:
    """Orientation von Mises: R0 + A * exp(kappa * cos(2*(theta - theta0)))."""
    return R0 + A * np.exp(kappa * np.cos(2.0 * (theta - theta0)))


def _vm_direction_model(
    theta: npt.NDArray,
    A_pref: float,
    A_null: float,
    kappa: float,
    theta0: float,
    baseline: float,
) -> np.ndarray:
    """Direction von Mises: two bumps at theta0 and theta0+pi.

    A_pref VM(theta0, kappa) + A_null VM(theta0+pi, kappa) + baseline,
    where VM(mu, kappa) = exp(kappa * cos(theta - mu)).
    """
    return (
        A_pref * np.exp(kappa * np.cos(theta - theta0))
        + A_null * np.exp(kappa * np.cos(theta - (theta0 + np.pi)))
        + baseline
    )


def _double_gaussian_model(
    theta: npt.NDArray,
    A1: float,
    A2: float,
    sigma: float,
    theta0: float,
    baseline: float,
) -> np.ndarray:
    """Double Gaussian: A1*G(theta0, sigma) + A2*G(theta0+pi, sigma) + b."""
    d1 = theta - theta0
    d2 = theta - (theta0 + np.pi)
    # Wrap to [-pi, pi]
    d1 = np.arctan2(np.sin(d1), np.cos(d1))
    d2 = np.arctan2(np.sin(d2), np.cos(d2))
    return (
        A1 * np.exp(-(d1**2) / (2 * sigma**2)) + A2 * np.exp(-(d2**2) / (2 * sigma**2)) + baseline
    )


def _vm_hwhh_deg(kappa: float, doubled: bool) -> float:
    """HWHH (degrees) of a von Mises bump.

    For the orientation form ``exp(kappa cos(2(θ-θ0)))`` the cosine
    runs over a full period in 180°, so the half-width on the
    underlying axis is half what you get for the direction form.

    Returns ``np.inf`` when *kappa* is too small for the bump to drop
    to half-height anywhere.
    """
    if abs(kappa) < 1e-10:
        return np.inf
    arg = 1.0 - np.log(2.0) / abs(kappa)
    if not -1.0 <= arg <= 1.0:
        return np.inf
    hwhh_rad = np.arccos(arg)
    if doubled:
        hwhh_rad *= 0.5
    return float(np.rad2deg(hwhh_rad))


def _nan_dict(keys: list[str]) -> dict:
    """Return a dict with all keys set to NaN."""
    return {k: np.nan for k in keys}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def goodness_of_fit(
    observed: npt.ArrayLike,
    predicted: npt.ArrayLike,
) -> float:
    r"""Coefficient of determination (R²).

    .. math::

        R^2 = 1 - \frac{\mathrm{SS_{res}}}{\mathrm{SS_{tot}}}

    Args:
        observed: Observed response values.
        predicted: Predicted (fitted) response values.

    Returns:
        R² value.  Returns ``np.nan`` if total variance is zero.
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def von_mises_fit(
    responses: npt.ArrayLike,
    angles_deg: npt.ArrayLike,
    *,
    tuning_type: TuningType = "orientation",
    return_fit: bool = False,
) -> dict | tuple[dict, np.ndarray]:
    r"""Fit a von Mises tuning curve.

    Two model variants are selected via *tuning_type*:

    *orientation*

    .. math::

        R(\theta) = R_0 + A\,\exp\!\bigl(\kappa\,\cos(2(\theta - \theta_0))\bigr)

    Single bump on the half-circle.  Use this when the data are
    sampled on 0–180°, or for direction data when only the axial
    selectivity matters.

    *direction*

    .. math::

        R(\theta) = A_\text{pref}\,e^{\kappa\cos(\theta - \theta_0)}
                  + A_\text{null}\,e^{\kappa\cos(\theta - (\theta_0+\pi))}
                  + b

    Two bumps at the preferred and null directions.  Use this when the
    data are sampled on 0–360° and you need to distinguish forward and
    reverse motion.

    Args:
        responses: Mean firing rates at each angle.
        angles_deg: Stimulus angles in degrees.
        tuning_type: ``"orientation"`` (default) or ``"direction"``.
        return_fit: If ``True``, also return the fitted curve array.

    Returns:
        Dict with these keys, regardless of ``tuning_type``:

        - ``tuning_type`` — echo of the parameter for downstream code
        - ``preferred_angle`` — preferred angle in degrees
          (``[0, 180)`` for orientation, ``[0, 360)`` for direction)
        - ``kappa`` — concentration of the von Mises bump
        - ``baseline`` — fitted offset
        - ``r_squared`` — coefficient of determination
        - ``bandwidth_hwhh`` — half-width at half-height of the
          preferred bump, in degrees on the original angle axis

        Direction fits also include:

        - ``amplitude_pref`` — peak height at the preferred direction
        - ``amplitude_null`` — peak height at the null direction
        - ``ds_ratio`` — ``(amp_pref − amp_null) / (amp_pref + amp_null)``,
          ``nan`` if both amplitudes are zero

        Orientation fits also include:

        - ``amplitude`` — single-bump amplitude

        On fit failure all numeric keys are ``nan``.

        When ``return_fit=True``, returns ``(dict, fitted_array)``.
    """
    if tuning_type not in ("orientation", "direction"):
        raise ValueError(f"tuning_type must be 'orientation' or 'direction', got {tuning_type!r}")

    common_keys = [
        "preferred_angle",
        "kappa",
        "baseline",
        "r_squared",
        "bandwidth_hwhh",
    ]
    if tuning_type == "orientation":
        keys = common_keys + ["amplitude"]
    else:
        keys = common_keys + ["amplitude_pref", "amplitude_null", "ds_ratio"]

    responses = np.asarray(responses, dtype=np.float64)
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    theta = np.deg2rad(angles_deg)

    try:
        pref_idx = int(np.argmax(responses))
        theta0_init = theta[pref_idx]
        A_init = float(np.max(responses) - np.min(responses))
        R0_init = float(np.min(responses))

        if tuning_type == "orientation":
            popt, _ = curve_fit(
                _vm_orientation_model,
                theta,
                responses,
                p0=[R0_init, A_init, 2.0, theta0_init],
                maxfev=10000,
            )
            R0, A, kappa, theta0 = popt
            fitted = _vm_orientation_model(theta, *popt)
            pref_deg = wrap180(np.rad2deg(theta0))
            extra = {"amplitude": float(A)}
            hwhh_deg = _vm_hwhh_deg(float(kappa), doubled=True)
        else:
            popt, _ = curve_fit(
                _vm_direction_model,
                theta,
                responses,
                p0=[A_init, A_init * 0.5, 2.0, theta0_init, R0_init],
                maxfev=10000,
            )
            A_pref, A_null, kappa, theta0, R0 = popt
            fitted = _vm_direction_model(theta, *popt)
            pref_deg = wrap360(np.rad2deg(theta0))
            denom = float(A_pref + A_null)
            ds_ratio = (float(A_pref) - float(A_null)) / denom if denom != 0 else np.nan
            extra = {
                "amplitude_pref": float(A_pref),
                "amplitude_null": float(A_null),
                "ds_ratio": ds_ratio,
            }
            hwhh_deg = _vm_hwhh_deg(float(kappa), doubled=False)

        r2 = goodness_of_fit(responses, fitted)
        result = {
            "tuning_type": tuning_type,
            "preferred_angle": pref_deg,
            "kappa": float(kappa),
            "baseline": float(R0),
            "r_squared": r2,
            "bandwidth_hwhh": hwhh_deg,
            **extra,
        }
    except (RuntimeError, ValueError, TypeError):
        result = {"tuning_type": tuning_type, **_nan_dict(keys)}
        fitted = np.full_like(responses, np.nan)

    if return_fit:
        return result, fitted
    return result


def double_gaussian_fit(
    responses: npt.ArrayLike,
    orientations_deg: npt.ArrayLike,
    return_fit: bool = False,
) -> dict | tuple[dict, np.ndarray]:
    r"""Fit a double Gaussian tuning curve.

    Model:

    .. math::

        R(\theta) = A_1\,G(\theta_0,\sigma) + A_2\,G(\theta_0+\pi,\sigma) + b

    Args:
        responses: Mean firing rates at each orientation.
        orientations_deg: Stimulus orientations in degrees.
        return_fit: If ``True``, also return the fitted curve array.

    Returns:
        Dict with keys: ``amp1``, ``amp2``, ``sigma`` (radians),
        ``theta0`` (degrees), ``baseline``, ``r_squared``.
    """
    keys = ["amp1", "amp2", "sigma", "theta0", "baseline", "r_squared"]
    responses = np.asarray(responses, dtype=np.float64)
    orientations_deg = np.asarray(orientations_deg, dtype=np.float64)
    theta = np.deg2rad(orientations_deg)

    try:
        pref_idx = int(np.argmax(responses))
        A_init = float(np.max(responses) - np.min(responses))
        popt, _ = curve_fit(
            _double_gaussian_model,
            theta,
            responses,
            p0=[A_init, A_init * 0.5, 0.5, theta[pref_idx], float(np.min(responses))],
            maxfev=10000,
        )
        A1, A2, sigma, theta0, baseline = popt
        fitted = _double_gaussian_model(theta, *popt)
        r2 = goodness_of_fit(responses, fitted)

        result = {
            "amp1": float(A1),
            "amp2": float(A2),
            "sigma": float(abs(sigma)),
            "theta0": wrap360(np.rad2deg(theta0)),
            "baseline": float(baseline),
            "r_squared": r2,
        }
    except (RuntimeError, ValueError, TypeError):
        result = _nan_dict(keys)
        fitted = np.full_like(responses, np.nan)

    if return_fit:
        return result, fitted
    return result


def tuning_curve_interpolation(
    responses: npt.ArrayLike,
    angles_deg: npt.ArrayLike,
    model: str = "von_mises_orientation",
) -> float:
    """Preferred angle from the peak of a fitted tuning curve.

    The fitted model is sampled across one *full* period (180° for
    orientation, 360° for direction / double-Gaussian) before taking
    the argmax.  Sampling only across the observed angle range would
    miss the true peak whenever the preferred angle sits across the
    wraparound — e.g. a direction-tuned cell with peak at 350° on data
    sampled at ``[0, 30, ..., 330]`` would erroneously return an angle
    inside ``[0, 330]``.

    Args:
        responses: Mean firing rates at each angle.
        angles_deg: Stimulus angles in degrees.
        model: Fitting model — ``"von_mises_orientation"``,
            ``"von_mises_direction"``, or ``"double_gaussian"``.

    Returns:
        Preferred angle in degrees, or ``np.nan`` on failure.  The
        result lies in ``[0, 180)`` for orientation models and in
        ``[0, 360)`` for direction / double-Gaussian models.
    """
    angles_deg = np.asarray(angles_deg, dtype=np.float64)

    if model == "von_mises_orientation":
        result, fitted = von_mises_fit(
            responses,
            angles_deg,
            tuning_type="orientation",
            return_fit=True,
        )
        if np.all(np.isnan(fitted)):
            return np.nan
        # Orientation model has period 180° on the angle axis.
        theta_fine = np.linspace(0.0, 180.0, 3600, endpoint=False)
        fine_fit = _vm_orientation_model(
            np.deg2rad(theta_fine),
            result["baseline"],
            result["amplitude"],
            result["kappa"],
            np.deg2rad(result["preferred_angle"]),
        )
    elif model == "von_mises_direction":
        result, fitted = von_mises_fit(
            responses,
            angles_deg,
            tuning_type="direction",
            return_fit=True,
        )
        if np.all(np.isnan(fitted)):
            return np.nan
        # Direction model has period 360°.
        theta_fine = np.linspace(0.0, 360.0, 3600, endpoint=False)
        fine_fit = _vm_direction_model(
            np.deg2rad(theta_fine),
            result["amplitude_pref"],
            result["amplitude_null"],
            result["kappa"],
            np.deg2rad(result["preferred_angle"]),
            result["baseline"],
        )
    elif model == "double_gaussian":
        result, fitted = double_gaussian_fit(
            responses,
            angles_deg,
            return_fit=True,
        )
        if np.all(np.isnan(fitted)):
            return np.nan
        # Double-Gaussian has two bumps at θ₀ and θ₀+π — period 360°.
        theta_fine = np.linspace(0.0, 360.0, 3600, endpoint=False)
        fine_fit = _double_gaussian_model(
            np.deg2rad(theta_fine),
            result["amp1"],
            result["amp2"],
            result["sigma"],
            np.deg2rad(result["theta0"]),
            result["baseline"],
        )
    else:
        raise ValueError(
            f"Unknown model {model!r}. Choose from "
            f"['von_mises_orientation', 'von_mises_direction', 'double_gaussian']"
        )

    return float(theta_fine[np.argmax(fine_fit)])
