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
    r"""Log-Gaussian: ``a * exp(-0.5*((log2(x)-mu)/sigma)^2) + b``.

    Callers must restrict ``x`` to **strictly positive** temporal
    frequencies.  TF=0 ("blank" / static-contrast trials) have no
    meaningful log-frequency representation and must be excluded
    upstream before passing ``x`` to this model — see
    :func:`temporal_frequency_tuning` for the contract.

    Convention: ``mu`` is in octaves (base-2 log of TF in Hz),
    matching Hawken et al. (1996) and Foster et al. (1985).
    """
    # Defensive only: callers strip TF=0 themselves, but a hard guard
    # makes any accidental call surface immediately rather than
    # silently extrapolating 40 octaves below μ.
    if np.any(np.asarray(x) <= 0):
        raise ValueError(
            "_log_gaussian requires strictly positive temporal "
            "frequencies; TF=0 (blank trials) must be removed by the "
            "caller before fitting.  See "
            "temporal_frequency_tuning() for the standard handling."
        )
    log_x = np.log2(x)
    return a * np.exp(-0.5 * ((log_x - mu) / sigma) ** 2) + b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def f1_phase(
    psth: npt.ArrayLike,
    fs: float,
    f_stim: float,
) -> float | np.ndarray:
    r"""Phase of the F1 (first harmonic) component.

    Extracts the phase at the stimulus temporal frequency from the FFT
    of the PSTH.  Works on 1-D and multi-dimensional arrays (last axis
    = time), following the same convention as
    :func:`~tuning.tuning.compute_f0_f1_f2`.

    **Phase convention.**

    - **Reference frame.** The PSTH is assumed to start at *trial
      onset* (t = 0).  If you align PSTHs to stimulus onset instead,
      the returned phase is shifted by ``-2π · f_stim · t_onset``;
      account for the offset when comparing absolute phases between
      analyses with different alignment.
    - **Sign / direction.** Returned via :func:`numpy.angle` on the
      output of :func:`scipy.fft.fft`, which uses the engineering /
      physics ``exp(-2π i f t)`` forward-transform convention.  A
      cosine ``cos(2π f_stim t)`` therefore has phase ``0``, and a
      sine ``sin(2π f_stim t)`` has phase ``-π/2``.  An increasing
      stimulus-aligned response (sin-like) yields a *negative* phase,
      a decreasing one yields a positive phase.
    - **Range.** ``(-π, π]`` radians, the standard branch cut of
      :func:`numpy.angle`.  ``+π`` and ``-π`` denote the same phase
      (the function returns ``+π`` exactly at the branch cut).
    - **Wraparound-safe averaging.** Phase is circular; arithmetic
      mean across cells / trials gives the wrong answer near
      ``±π``.  Use the resultant length
      ``|mean(exp(1j·phase))|`` for consistency and
      :func:`~neural_cca._utils.circ_mean` (period ``2π``) for the
      mean phase.  This is exactly what
      :func:`trial_to_trial_reliability(stat="f1_phase")` does.

    Args:
        psth: Firing-rate time series.  Shape ``(time,)`` or
            ``(..., time)``.
        fs: Sampling frequency of the PSTH in Hz (= 1 / bin_size).
        f_stim: Stimulus temporal frequency in Hz.

    Returns:
        Phase in radians in ``(-π, π]`` (float for 1-D input, array
        otherwise).

    References:
        Skottun, B. C., De Valois, R. L., Grosof, D. H., Movshon,
        J. A., Albrecht, D. G. & Bonds, A. B. (1991).  *Classifying
        simple and complex cells on the basis of response modulation*.
        Vision Research 31(7), 1079–1086.
        doi:10.1016/0042-6989(91)90033-2.

        Berens, P. (2009).  *CircStat: a MATLAB toolbox for circular
        statistics*.  Journal of Statistical Software 31(10), 1–21.
        doi:10.18637/jss.v031.i10.

        SciPy FFT phase / sign convention:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html
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

        - ``"temporal_freqs"`` — unique TFs tested (includes 0 if present)
        - ``"amplitudes"`` — response amplitude per TF (same order)
        - ``"preferred_tf"`` — TF with strongest response (may be 0
          for cells with the strongest response on blank trials)
        - ``"bandwidth"`` — HWHH of log-Gaussian fit (octaves), or
          ``np.inf`` on failure.  The fit is computed on
          *positive-TF* samples only because log₂(0) is undefined;
          TF=0 trials are reported separately via ``baseline_response``.
        - ``"r_squared"`` — goodness of fit, or ``np.nan`` on failure
        - ``"fit_curve"`` — fitted values at each unique TF, with
          ``np.nan`` at TF=0 (the log-Gaussian is undefined there),
          or ``None`` if the fit failed.
        - ``"baseline_response"`` — mean amplitude on TF=0 (blank /
          static-contrast) trials, or ``np.nan`` if no TF=0 sample is
          present.  Reported separately so it does not contaminate
          ``bandwidth`` / ``preferred_tf`` via log-frequency
          extrapolation.

    References:
        Foster, K. H., Gaska, J. P., Nagler, M. & Pollen, D. A.
        (1985).  *Spatial and temporal frequency selectivity of
        neurones in visual cortical areas V1 and V2 of the macaque
        monkey*.  Journal of Physiology 365, 331–363.

        Hawken, M. J., Shapley, R. M. & Grosof, D. H. (1996).
        *Temporal-frequency selectivity in monkey visual cortex*.
        Visual Neuroscience 13, 477–492.
        doi:10.1017/S0952523800008154.
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

    # Blank (TF=0) trials are categorically different from low-TF
    # gratings — they have no log-frequency representation.  Strip
    # them from the *fit* and surface their mean response separately
    # as ``baseline_response`` so callers can inspect the spontaneous
    # / blank level without contaminating ``bandwidth`` / ``preferred_tf``.
    # See Hawken et al. (1996, Vis. Neurosci. 13:477) and Foster et al.
    # (1985, J. Physiol. 365:331) for the convention.
    positive_mask = unique_tfs > 0
    fit_tfs = unique_tfs[positive_mask]
    fit_amps = amplitudes[positive_mask]
    if (~positive_mask).any():
        baseline_response = float(np.mean(amplitudes[~positive_mask]))
    else:
        baseline_response = np.nan

    # Fit log-Gaussian on the positive-TF subset only.
    fit_curve = None
    bandwidth = np.inf
    r_squared = np.nan
    if len(fit_tfs) >= 4 and np.any(fit_amps > 0):
        try:
            a0 = float(np.max(fit_amps))
            pref_positive_idx = int(np.argmax(fit_amps))
            mu0 = float(np.log2(fit_tfs[pref_positive_idx]))
            popt, _ = curve_fit(
                _log_gaussian,
                fit_tfs,
                fit_amps,
                p0=[a0, mu0, 1.0, 0.0],
                maxfev=5000,
            )
            # Evaluate the curve on the full (unique_tfs) grid so the
            # return shape is consistent; sample positions at TF=0 are
            # masked to NaN because the log-Gaussian is undefined there.
            fit_curve = np.full_like(unique_tfs, np.nan, dtype=np.float64)
            fit_curve[positive_mask] = _log_gaussian(fit_tfs, *popt)
            ss_res = float(np.sum((fit_amps - fit_curve[positive_mask]) ** 2))
            ss_tot = float(np.sum((fit_amps - np.mean(fit_amps)) ** 2))
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
        "baseline_response": baseline_response,
    }
