"""Temporal frequency tuning and F1 phase extraction.

Provides functions for analysing temporal aspects of visual responses:
temporal-frequency tuning curves and the phase of the first harmonic (F1).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

__all__ = [
    "temporal_frequency_tuning",
    "f1_phase",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_gaussian(x: npt.NDArray, a: float, mu: float, sigma: float, b: float) -> np.ndarray:
    """Log-Gaussian: a * exp(-0.5*((log2(x)-mu)/sigma)^2) + b."""
    log_x = np.log2(np.maximum(x, 1e-12))
    return a * np.exp(-0.5 * ((log_x - mu) / sigma) ** 2) + b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def f1_phase(
    psth: npt.ArrayLike,
    fs: float,
    f_stim: float,
) -> float | np.ndarray:
    """Phase of the F1 (first harmonic) component.

    Extracts the phase at the stimulus temporal frequency from the FFT
    of the PSTH.  Works on 1-D and multi-dimensional arrays (last axis
    = time), following the same convention as
    :func:`~tuning.tuning.compute_f0_f1_f2`.

    Args:
        psth: Firing-rate time series.  Shape ``(time,)`` or
            ``(..., time)``.
        fs: Sampling frequency of the PSTH in Hz (= 1 / bin_size).
        f_stim: Stimulus temporal frequency in Hz.

    Returns:
        Phase in radians (float for 1-D input, array otherwise).
    """
    psth = np.asarray(psth, dtype=np.float64)
    N = psth.shape[-1]
    fft_vals = fft(psth, axis=-1)
    freqs = fftfreq(N, d=1.0 / fs)
    idx1 = int(np.argmin(np.abs(freqs - f_stim)))
    phase = np.angle(fft_vals[..., idx1])
    if phase.ndim == 0:
        return float(phase)
    return phase


def temporal_frequency_tuning(
    spike_times: npt.NDArray[np.float64],
    trials: npt.NDArray[np.int64],
    temporal_freqs: npt.NDArray[np.float64],
    bin_size: float = 0.05,
    stim_window: tuple[float, float] = (0.5, 2.5),
    response_metric: str = "f1",
    cluster_labels: npt.NDArray[np.int64] | None = None,
    cluster_id: int | None = None,
) -> dict:
    """Temporal-frequency (TF) tuning curve.

    Groups trials by temporal frequency, computes the response at each
    frequency, and optionally fits a log-Gaussian.

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        temporal_freqs: TF value per trial (Hz).
        bin_size: PSTH bin width (seconds).
        stim_window: ``(onset, end)`` of the stimulus period within
            each trial (seconds). Only spikes inside this window are
            counted; the per-trial PSTH is built on the same window.
        response_metric: ``"f1"`` for F1 amplitude, ``"mfr"`` for mean
            firing rate.
        cluster_labels: Cluster label per spike (optional).
        cluster_id: Cluster ID for per-cluster analysis.

    Returns:
        Dict with keys:

        - ``"temporal_freqs"`` — unique TFs tested
        - ``"amplitudes"`` — response amplitude per TF
        - ``"preferred_tf"`` — TF with strongest response
        - ``"bandwidth"`` — HWHH of log-Gaussian fit (octaves), or
          ``np.inf`` on failure
        - ``"r_squared"`` — goodness of fit, or ``np.nan`` on failure
        - ``"fit_curve"`` — fitted values at each TF, or ``None``
    """
    spike_times = np.asarray(spike_times, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.int64)
    temporal_freqs = np.asarray(temporal_freqs, dtype=np.float64)

    # Filter by cluster
    if cluster_id is not None and cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        trials = trials[mask]

    s_on, s_end = stim_window
    stim_dur = s_end - s_on
    unique_tfs = np.unique(temporal_freqs)
    amplitudes = np.zeros(len(unique_tfs))

    for i, tf in enumerate(unique_tfs):
        tf_trials = np.where(temporal_freqs == tf)[0]
        tf_spikes = []
        for t in tf_trials:
            t_mask = trials == t
            spk = spike_times[t_mask]
            tf_spikes.append(spk[(spk > s_on) & (spk <= s_end)])

        if response_metric == "mfr":
            total = sum(len(s) for s in tf_spikes)
            amplitudes[i] = total / (len(tf_trials) * stim_dur) if len(tf_trials) > 0 else 0.0
        else:
            # F1 amplitude: build PSTH per trial, compute F1, average.
            # ``ceil + linspace`` over ``[s_on, s_end]`` so the bin
            # count is stable regardless of float rounding.
            psth_fs = 1.0 / bin_size
            n_bins = int(np.ceil((s_end - s_on) / bin_size))
            bins = np.linspace(s_on, s_end, n_bins + 1)
            f1_vals = []
            for spk in tf_spikes:
                if len(spk) == 0:
                    f1_vals.append(0.0)
                else:
                    counts, _ = np.histogram(spk, bins=bins)
                    rate = counts / bin_size
                    N = len(rate)
                    fft_v = fft(rate)
                    freqs = fftfreq(N, d=1.0 / psth_fs)
                    idx1 = int(np.argmin(np.abs(freqs - tf)))
                    f1_vals.append(2.0 * float(np.abs(fft_v[idx1])) / N)
            amplitudes[i] = float(np.mean(f1_vals)) if f1_vals else 0.0

    # Find preferred TF
    pref_idx = int(np.argmax(amplitudes))
    preferred_tf = float(unique_tfs[pref_idx])

    # Fit log-Gaussian
    fit_curve = None
    bandwidth = np.inf
    r_squared = np.nan
    if len(unique_tfs) >= 4 and np.any(amplitudes > 0):
        try:
            a0 = float(np.max(amplitudes))
            mu0 = float(np.log2(preferred_tf)) if preferred_tf > 0 else 0.0
            popt, _ = curve_fit(
                _log_gaussian,
                unique_tfs,
                amplitudes,
                p0=[a0, mu0, 1.0, 0.0],
                maxfev=5000,
            )
            fit_curve = _log_gaussian(unique_tfs, *popt)
            ss_res = float(np.sum((amplitudes - fit_curve) ** 2))
            ss_tot = float(np.sum((amplitudes - np.mean(amplitudes)) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            bandwidth = float(abs(popt[2]) * np.sqrt(2 * np.log(2)))
        except (RuntimeError, ValueError):
            pass

    return {
        "temporal_freqs": unique_tfs,
        "amplitudes": amplitudes,
        "preferred_tf": preferred_tf,
        "bandwidth": bandwidth,
        "r_squared": r_squared,
        "fit_curve": fit_curve,
    }
