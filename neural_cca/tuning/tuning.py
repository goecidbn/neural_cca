"""Tuning curve analysis and harmonic decomposition.

Provides tuning bandwidth estimation (HWHH of Gaussian fit),
F0/F1/F2 harmonic extraction from PSTHs, preferred orientation via
the vector-sum method, and the composite ``get_os_metrics`` function.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

from .._utils import circ_dist, circ_mean, guarded_divide, make_rng
from ._filter import _build_trial_filter
from .fitting import (
    double_gaussian_fit as _dg_fit,
)
from .fitting import (
    von_mises_fit as _vm_fit,
)
from .selectivity import (
    circular_variance,
    dosi_circular_normalised,
    dsi_two_point,
    gdsi,
    gosi,
    osi_two_point,
)
from .statistics import (
    anova_across_orientations as _anova,
)
from .statistics import (
    bootstrap_ci_strata as _boot_strata,
)

__all__ = [
    "tuning_bandwidth",
    "compute_f0_f1_f2",
    "preferred_dori",
    "get_os_metrics",
    "OsMetricsResult",
]


class OsMetricsResult(TypedDict, total=False):
    """Typed schema of the dict returned by :func:`get_os_metrics`.

    All keys are lower-case ``snake_case``.  Every entry is optional
    (``total=False``) because the actual key set depends on
    ``return_verbose``, ``stim_frequency``, ``compute_gosi``,
    ``compute_p_values``, ``fit_model`` and ``bootstrap_ci``.

    Use this for type-checker support and as the single source of
    truth for the canonical key spelling — there is no implicit
    ``CamelCase`` / ``Title Case`` rendering anywhere in the package.
    """

    # Core selectivity (always present)
    osi: float
    dsi: float
    circular_variance: float
    tuning_bandwidth: float
    preferred_orientation: float
    preferred_direction: float
    closest_orientation: float

    # F0 / F1 / F2 harmonic summary (return_verbose >= 1)
    f0_mean: float
    f1_mean: float
    f2_mean: float
    f1_f0_mean: float
    f2_f0_mean: float
    f2_f1_mean: float

    # Verbose intermediate arrays (return_verbose == 2)
    spike_times_by_trial: dict
    mfrs: np.ndarray
    psth_by_trial: dict
    f0: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f1_f0: np.ndarray
    f2_f0: np.ndarray
    f2_f1: np.ndarray

    # gOSI / gDSI (compute_gosi=True) — vector-sum / "global" forms.
    gosi: float
    gdsi: float
    # Two-point Niell & Stryker (2008) forms (compute_gosi=True).
    osi_two_point: float
    dsi_two_point: float

    # Significance (compute_p_values=True)
    osi_p_value: float
    dsi_p_value: float
    gosi_p_value: float
    gdsi_p_value: float
    anova_p_value: float

    # Fit results (fit_model not None)
    fit_model: str
    fit_preferred_angle: float
    fit_r_squared: float
    fit_bandwidth: float

    # Bootstrap CIs (bootstrap_ci=True). Each value is a dict with
    # snake_case keys: ``estimate``, ``ci_lower``, ``ci_upper``, ``se``.
    osi_ci: dict
    dsi_ci: dict
    gosi_ci: dict
    gdsi_ci: dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gauss(x: npt.ArrayLike, a: float, x0: float, sigma: float, b: float) -> np.ndarray:
    """Gaussian with baseline: a * exp(-(x-x0)^2 / (2*sigma^2)) + b."""
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tuning_bandwidth(
    responses: npt.NDArray[np.float64],
    orientations: npt.NDArray[np.float64],
    *,
    method: str = "von_mises",
) -> float:
    r"""Half-width at half-height (HWHH) of an orientation tuning curve.

    By default (``method="von_mises"``, *recommended*) the bandwidth is
    derived from a **circular von Mises fit** in orientation space
    (period 180°), which is the empirically best-fitting parametric
    model for V1 orientation tuning (Swindale 1998).  The von Mises
    handles wraparound correctly, so a cell with preferred orientation
    near 0° or 180° gets the same HWHH it would at 90°.

    The legacy linear-Gaussian path remains available via
    ``method="gaussian"`` for reproducing older results, but it is
    **biased** for preferred orientations near 0°/180°: the Gaussian
    has no periodic extension, so half the tuning bump is truncated at
    the angle-axis boundary and ``curve_fit`` underestimates the
    width.  The bias is systematic with preferred orientation and
    can contaminate population HWHH distributions; do not use this
    method when reporting bandwidth statistics across a population.

    Args:
        responses: Firing rates at each orientation.
        orientations: Orientations in degrees.
        method: ``"von_mises"`` (default, recommended) — circular fit
            via :func:`~neural_cca.tuning.fitting.von_mises_fit` with
            ``tuning_type="orientation"``.  ``"gaussian"`` — legacy
            linear Gaussian fit (biased near 0°/180°, retained for
            backwards compatibility).

    Returns:
        HWHH in degrees, or ``np.inf`` if the fit fails or responses
        are flat.

    References:
        Swindale, N. V. (1998).  *Orientation tuning curves: empirical
        description and estimation of parameters*.  Biological
        Cybernetics 78(1), 45–56.  doi:10.1007/s004220050411.

        Mazurek, M., Kager, M. & Van Hooser, S. D. (2014).  *Robust
        quantification of orientation selectivity and direction
        selectivity*.  Frontiers in Neural Circuits 8:92.
        doi:10.3389/fncir.2014.00092.
    """
    if method == "von_mises":
        # Delayed import keeps the top-level import graph tidy
        # (``fitting`` already imports from this module's siblings).
        from .fitting import von_mises_fit

        if np.allclose(responses, responses[0]):
            return np.inf
        fit = von_mises_fit(responses, orientations, tuning_type="orientation")
        hwhh = fit["bandwidth_hwhh"]
        # ``bandwidth_hwhh`` is NaN on fit failure; surface ``inf`` for
        # backwards compatibility with the legacy contract.
        return float(hwhh) if np.isfinite(hwhh) else np.inf

    if method == "gaussian":
        if np.allclose(responses, responses[0]):
            return np.inf
        a0 = np.max(responses) - np.min(responses)
        x0 = orientations[np.argmax(responses)]
        sigma0 = 20.0
        b0 = np.min(responses)
        try:
            popt, _ = curve_fit(_gauss, orientations, responses, p0=[a0, x0, sigma0, b0])
        except RuntimeError:
            return np.inf
        sigma = abs(popt[2])
        return float(sigma * np.sqrt(2 * np.log(2)))

    raise ValueError(
        f"Unknown method {method!r}.  Choose 'von_mises' (default, "
        "recommended) or 'gaussian' (legacy, biased near 0°/180°)."
    )


def compute_f0_f1_f2(
    psth: npt.ArrayLike,
    fs: float,
    f_stim: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract F0, F1, F2 harmonic components from a PSTH.

    Works on 1-D and multi-dimensional PSTHs (last axis = time).

    Args:
        psth: Firing-rate time series.  Can be ``(time,)`` or
            ``(..., time)``.
        fs: Sampling frequency of the PSTH in Hz (= 1 / bin_size).
        f_stim: Stimulus temporal frequency in Hz.

    Returns:
        ``(F0, F1, F2)`` — each with shape ``psth.shape[:-1]``.
    """
    psth = np.asarray(psth, dtype=np.float64)
    N = psth.shape[-1]
    F0 = np.mean(psth, axis=-1)

    fft_vals = fft(psth, axis=-1)
    freqs = fftfreq(N, d=1.0 / fs)

    idx1 = int(np.argmin(np.abs(freqs - f_stim)))
    idx2 = int(np.argmin(np.abs(freqs - 2 * f_stim)))

    F1 = 2.0 * np.abs(fft_vals[..., idx1]) / N
    F2 = 2.0 * np.abs(fft_vals[..., idx2]) / N

    return F0, F1, F2


def preferred_dori(
    responses: npt.ArrayLike,
    orientations_deg: npt.ArrayLike,
    direction_selectivity: bool = False,
) -> float:
    """Preferred orientation (or direction) via the vector-sum method.

    Thin wrapper around :func:`neural_cca._utils.circ_mean` that
    selects the right circular period for orientation vs direction
    statistics.

    Args:
        responses: Mean firing rates R(theta) at each orientation.
        orientations_deg: Stimulus orientations in degrees.
        direction_selectivity: If ``True`` return preferred direction
            (0–360°); if ``False`` return preferred orientation
            (0–180°).

    Returns:
        Preferred orientation in degrees.
    """
    period = 360.0 if direction_selectivity else 180.0
    return circ_mean(orientations_deg, weights=responses, period=period)


def get_os_metrics(
    spike_times: npt.NDArray[np.float64],
    trials: npt.NDArray[np.int64],
    angles: npt.NDArray[np.float64],
    bin_size: float = 0.05,
    all_clusters: bool = True,
    cluster_labels: npt.NDArray[np.int64] | None = None,
    cluster_id: int | None = None,
    # NOTE: Natal-specific default; v0.2.0 will make this required.
    stim_window: tuple[float, float] = (0.5, 2.5),
    # NOTE: Natal-specific default; v0.2.0 will make this required.
    stim_frequency: float | None = 2.0,
    return_verbose: int = 1,
    fit_model: str | None = None,
    compute_gosi: bool = True,
    compute_p_values: bool = False,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> OsMetricsResult:
    """Orientation-selectivity metrics for a set of spike times.

    Args:
        spike_times: Spike times (trial-relative, in seconds).
        trials: Trial index for each spike.
        angles: Stimulus angle (degrees) for each trial.
        bin_size: PSTH bin width in seconds.
        all_clusters: Use all spikes (``True``) or filter by
            *cluster_id*.
        cluster_labels: Cluster label per spike (required when
            ``all_clusters=False``).
        cluster_id: Cluster ID to select (required when
            ``all_clusters=False``).
        stim_window: ``(onset, end)`` of the stimulus period within
            each trial (seconds). Spikes inside this half-open
            interval ``(onset, end]`` are counted; the per-trial PSTH
            is built on the same window.
        stim_frequency: Temporal frequency of the stimulus (Hz).
            Set to ``None`` to skip F0/F1/F2 computation.
        return_verbose: 0 = core metrics only; 1 = core + F0/F1/F2
            summary (NaN when skipped); 2 = also intermediate arrays.
        fit_model: Tuning-curve model to fit — ``"von_mises_orientation"``,
            ``"von_mises_direction"``, or ``"double_gaussian"``.
            ``None`` to skip fitting.
        compute_gosi: If ``True`` (default), add gOSI and gDSI.
        compute_p_values: If ``True``, add Rayleigh *p*-values for
            OSI, DSI, gOSI, gDSI and an ANOVA *p*-value.
        bootstrap_ci: If ``True``, add bootstrap confidence intervals
            for OSI, DSI, gOSI, gDSI.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g. 0.95 for 95 % CI).
        rng: ``numpy.random.Generator``, integer seed, or ``None``.
            A *single* generator is shared across the OSI/DSI/gOSI/gDSI
            bootstraps so the full result is reproducible from one
            seed.  ``None`` (default) → fresh, unseeded generator.

    Returns:
        Dictionary of metric names to values.

    Raises:
        ValueError: If ``cluster_id`` or ``cluster_labels`` are missing
            when ``all_clusters=False``.
    """
    if not all_clusters and cluster_id is None:
        raise ValueError("cluster_id must be specified when all_clusters is False.")
    if not all_clusters and cluster_labels is None:
        raise ValueError("cluster_labels must be provided when all_clusters is False.")

    s_on, s_end = stim_window

    # Build the per-trial spike filter exactly once.  Every downstream
    # consumer (ANOVA, modulation ratio, F0/F1/F2 binning, the
    # selectivity stats below) reads from this object instead of
    # re-deriving its own filter from spike_times / trials / labels.
    # The private ``_filter=`` keyword on each helper short-circuits
    # the rebuild path.
    trial_filter = _build_trial_filter(
        spike_times,
        trials,
        angles,
        stim_window=stim_window,
        cluster_labels=cluster_labels if not all_clusters else None,
        cluster_id=cluster_id if not all_clusters else None,
    )
    spike_times_by_trial = trial_filter.spike_times_by_trial
    mfrs = trial_filter.mfrs
    angles = trial_filter.angles

    # --- Core orientation metrics ---
    pref_ori = preferred_dori(mfrs, angles)
    # ``pref_ori`` lives in [0, 180) (orientation space) but ``angles``
    # may span [0, 360); linear ``np.abs(angles - pref_ori)`` would pick
    # the wrong sample whenever the preferred orientation sits near the
    # 0°/180° wraparound.  Use circular distance with period 180 to find
    # the nearest *orientation* sample regardless of wraparound.
    closest_idx = int(np.argmin(circ_dist(angles, pref_ori, period=180.0)))

    _r: dict = {
        "osi": dosi_circular_normalised(mfrs, angles),
        "dsi": dosi_circular_normalised(mfrs, angles, direction_selectivity=True),
        "circular_variance": circular_variance(mfrs, angles),
        "tuning_bandwidth": tuning_bandwidth(mfrs, angles),
        "preferred_orientation": pref_ori,
        "preferred_direction": preferred_dori(
            mfrs,
            angles,
            direction_selectivity=True,
        ),
        "closest_orientation": float(angles[closest_idx]),
    }

    if return_verbose == 2:
        _r["spike_times_by_trial"] = spike_times_by_trial
        _r["mfrs"] = mfrs

    # --- F0 / F1 / F2 harmonics ---
    # These require ``stim_frequency``; when it is ``None`` the harmonic
    # keys are filled with NaN but **execution continues** so gOSI,
    # p-values, fits, and bootstrap CIs still run — none of them depend
    # on harmonics.  (An earlier version returned early here, which
    # silently skipped all the post-harmonic blocks.)
    if stim_frequency is None:
        if return_verbose >= 1:
            for k in ("f0_mean", "f1_mean", "f2_mean", "f1_f0_mean", "f2_f0_mean", "f2_f1_mean"):
                _r[k] = np.nan
    else:
        psth_fs = 1.0 / bin_size
        # ``ceil + linspace`` so the bin count is stable regardless of
        # floating-point rounding of ``(s_end - s_on) / bin_size``.
        n_bins = int(np.ceil((s_end - s_on) / bin_size))
        bins = np.linspace(s_on, s_end, n_bins + 1)

        psth_by_trial: dict[int, npt.NDArray] = {}
        for trial_idx, spikes in spike_times_by_trial.items():
            if len(spikes) == 0:
                psth_by_trial[trial_idx] = np.zeros(len(bins) - 1)
            else:
                counts, _ = np.histogram(spikes, bins=bins)
                psth_by_trial[trial_idx] = counts / bin_size

        _F0, _F1, _F2 = compute_f0_f1_f2(list(psth_by_trial.values()), psth_fs, stim_frequency)

        if return_verbose >= 1:
            _r.update(
                {
                    "f0_mean": float(_F0.mean()),
                    "f1_mean": float(_F1.mean()),
                    "f2_mean": float(_F2.mean()),
                    "f1_f0_mean": guarded_divide(_F1.mean(), _F0.mean()),
                    "f2_f0_mean": guarded_divide(_F2.mean(), _F0.mean()),
                    "f2_f1_mean": guarded_divide(_F2.mean(), _F1.mean()),
                }
            )

        if return_verbose == 2:
            _r.update(
                {
                    "psth_by_trial": psth_by_trial,
                    "f0": _F0,
                    "f1": _F1,
                    "f2": _F2,
                    "f1_f0": guarded_divide(_F1, _F0),
                    "f2_f0": guarded_divide(_F2, _F0),
                    "f2_f1": guarded_divide(_F2, _F1),
                }
            )

    # --- v0.3.0 extensions ---

    # gOSI / gDSI (vector-sum / "global") and the two-point variants.
    # Both families are reported when ``compute_gosi=True`` because a
    # methods-section reader can pick whichever convention their
    # reference paper uses without re-running the analysis — and the
    # two numbers diverge whenever the tuning curve is not a single
    # cosine bump.  See ``selectivity.py`` module docstring for the
    # naming convention.
    if compute_gosi:
        _r["gosi"] = gosi(mfrs, angles)
        _r["gdsi"] = gdsi(mfrs, angles)
        _r["osi_two_point"] = osi_two_point(mfrs, angles)
        _r["dsi_two_point"] = dsi_two_point(mfrs, angles)

    # P-values via Rayleigh test (+ ANOVA)
    if compute_p_values:
        osi_d = dosi_circular_normalised(mfrs, angles, p_value=True)
        dsi_d = dosi_circular_normalised(mfrs, angles, direction_selectivity=True, p_value=True)
        _r["osi_p_value"] = osi_d["p_value"]
        _r["dsi_p_value"] = dsi_d["p_value"]
        if compute_gosi:
            gosi_d = gosi(mfrs, angles, p_value=True)
            gdsi_d = gdsi(mfrs, angles, p_value=True)
            _r["gosi_p_value"] = gosi_d["p_value"]
            _r["gdsi_p_value"] = gdsi_d["p_value"]
        anova_result = _anova(
            spike_times,
            trials,
            angles,
            stim_window=stim_window,
            cluster_labels=cluster_labels if not all_clusters else None,
            cluster_id=cluster_id if not all_clusters else None,
            _filter=trial_filter,
        )
        _r["anova_p_value"] = anova_result["p_value"]

    # Tuning curve fitting
    if fit_model is not None:
        if fit_model == "von_mises_orientation":
            fit_result = _vm_fit(mfrs, angles, tuning_type="orientation")
            _r["fit_preferred_angle"] = fit_result["preferred_angle"]
        elif fit_model == "von_mises_direction":
            fit_result = _vm_fit(mfrs, angles, tuning_type="direction")
            _r["fit_preferred_angle"] = fit_result["preferred_angle"]
        elif fit_model == "double_gaussian":
            fit_result = _dg_fit(mfrs, angles)
            _r["fit_preferred_angle"] = fit_result.get("theta0", np.nan)
        else:
            raise ValueError(
                f"Unknown fit_model {fit_model!r}. Choose from "
                f"['von_mises_orientation', 'von_mises_direction', 'double_gaussian']"
            )
        _r["fit_model"] = fit_model
        _r["fit_r_squared"] = fit_result.get("r_squared", np.nan)
        _r["fit_bandwidth"] = fit_result.get("bandwidth_hwhh", fit_result.get("sigma", np.nan))

    # Bootstrap CIs — stratified by stimulus angle so each iteration
    # preserves the (rate, angle) pairing. A plain bootstrap over
    # `mfrs` would shuffle trials across angles and produce a
    # meaningless null for the orientation/direction-selectivity
    # statistics computed below.
    #
    # A single ``numpy.random.Generator`` is materialised here and
    # passed to *every* bootstrap call.  Successive calls advance the
    # same stream, so the full result is reproducible from one seed
    # (or freshly random when ``rng=None``).
    if bootstrap_ci:
        boot_rng = make_rng(rng)

        def _osi_func(rates: np.ndarray, ang: np.ndarray) -> float:
            return dosi_circular_normalised(rates, ang)

        def _dsi_func(rates: np.ndarray, ang: np.ndarray) -> float:
            return dosi_circular_normalised(rates, ang, direction_selectivity=True)

        _r["osi_ci"] = _boot_strata(
            mfrs,
            angles,
            _osi_func,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            rng=boot_rng,
        )
        _r["dsi_ci"] = _boot_strata(
            mfrs,
            angles,
            _dsi_func,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            rng=boot_rng,
        )
        if compute_gosi:

            def _gosi_func(rates: np.ndarray, ang: np.ndarray) -> float:
                return gosi(rates, ang)

            def _gdsi_func(rates: np.ndarray, ang: np.ndarray) -> float:
                return gdsi(rates, ang)

            _r["gosi_ci"] = _boot_strata(
                mfrs,
                angles,
                _gosi_func,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                rng=boot_rng,
            )
            _r["gdsi_ci"] = _boot_strata(
                mfrs,
                angles,
                _gdsi_func,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                rng=boot_rng,
            )

    return _r
