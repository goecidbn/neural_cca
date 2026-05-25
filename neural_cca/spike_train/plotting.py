"""Spike train visualisation helpers.

Provides ISI histograms, waveform snippet overlays, spike raster
plots, autocorrelograms, PSTHs, firing-rate stability, first-spike
latency, and trial reliability heatmaps.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .analysis import (
    AcgNormalize,
)
from .analysis import (
    autocorrelogram as _acg,
)
from .analysis import (
    firing_rate_stability as _frs,
)
from .analysis import (
    first_spike_latency as _fsl,
)
from .analysis import (
    psth as _psth,
)

__all__ = [
    "plot_isi_histogram",
    "plot_waveform_snippets",
    "plot_spike_raster",
    "plot_autocorrelogram",
    "plot_psth",
    "plot_firing_rate_stability",
    "plot_first_spike_latency",
    "plot_trial_reliability_heatmap",
]


def plot_isi_histogram(
    spike_times: npt.NDArray[np.float64],
    trials: npt.NDArray | None = None,
    bin_max: float = 0.1,
    bin_width: float = 0.001,
    refractory_period: float = 0.001,
    ax: plt.Axes | None = None,
    color: str = "steelblue",
) -> plt.Axes:
    """Plot the inter-spike interval distribution.

    For trial-based recordings, pass *trials* so the ISIs are computed
    *within each trial only*.  Without it, the function falls back to
    ``np.diff(spike_times)`` and filters negative values — which is
    only correct when the spike train is ordered trial-by-trial (not
    globally sorted by trial-relative time).  See
    :func:`~neural_cca.spike_train.analysis.autocorrelogram` for
    the full description of the trial-relative bug class.

    Args:
        spike_times: Spike times in seconds.
        trials: Optional trial index per spike.  When given, ISIs are
            computed within each trial only.
        bin_max: Maximum ISI to display (seconds).
        bin_width: Histogram bin width (seconds).
        refractory_period: Refractory period (seconds).  Shown as a
            vertical dashed line.
        ax: Existing ``Axes`` (created if ``None``).
        color: Histogram fill colour.

    Returns:
        ``Axes`` with the plot.
    """
    # ``_per_trial_isis`` is an underscore-private helper and is
    # therefore deliberately *not* in the module-top ``from .analysis
    # import (...)`` block.  Pull it in locally rather than promoting
    # an internal name into this module's namespace.
    from .analysis import _per_trial_isis

    per_trial = _per_trial_isis(spike_times, trials)
    isis = np.concatenate(per_trial) if per_trial else np.empty(0)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4), dpi=150)

    n_bins = max(1, int(np.ceil(bin_max / bin_width)))
    ax.hist(
        isis,
        range=(0, bin_max),
        bins=n_bins,
        color=color,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.axvline(
        refractory_period,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"refr. period ({refractory_period * 1e3:.1f} ms)",
    )
    ax.set_xlabel("ISI (s)")
    ax.set_ylabel("Count")
    ax.set_title("Inter-Spike Interval Distribution")
    ax.legend(fontsize="small")
    return ax


def plot_waveform_snippets(
    waveforms: npt.NDArray[np.float64],
    waveform_fs: float = 32_000.0,
    invert: bool = True,
    ax: plt.Axes | None = None,
    color: str = "black",
    alpha: float = 0.1,
) -> plt.Axes:
    """Overlay of waveform snippets with the mean waveform highlighted.

    Args:
        waveforms: (n_spikes, snippet_length) waveform matrix.
        waveform_fs: Sampling rate of waveforms (Hz).
        invert: If ``True`` plot ``-waveform`` (extracellular
            convention).
        ax: Existing ``Axes`` (created if ``None``).
        color: Colour for individual traces.
        alpha: Transparency for individual traces.

    Returns:
        ``Axes`` with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4), dpi=150)

    snippet_length = waveforms.shape[1]
    time_ms = np.arange(snippet_length) / waveform_fs * 1000.0
    sign = -1.0 if invert else 1.0

    ax.plot(time_ms, sign * waveforms.T, color=color, alpha=alpha, linewidth=0.5, rasterized=True)
    ax.plot(time_ms, sign * np.mean(waveforms, axis=0), color="red", linewidth=2, label="Mean")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"Spike Waveforms (n={len(waveforms)})")
    ax.legend(fontsize="small")
    return ax


def plot_spike_raster(
    spike_times: npt.NDArray[np.float64],
    trials: npt.NDArray[np.int64],
    stim_onset: float | None = None,
    ax: plt.Axes | None = None,
    color: str = "black",
    marker_size: float = 1.0,
) -> plt.Axes:
    """Spike raster plot (one row per trial).

    Args:
        spike_times: Spike times in seconds (trial-relative).
        trials: Trial index per spike.
        stim_onset: If provided, draw a vertical dashed line at
            stimulus onset.
        ax: Existing ``Axes`` (created if ``None``).
        color: Marker colour.
        marker_size: Size of each spike marker.

    Returns:
        ``Axes`` with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.scatter(spike_times, trials, marker="|", s=marker_size, color=color, linewidth=0.5)
    if stim_onset is not None:
        ax.axvline(
            stim_onset,
            color="red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
            label="Stimulus onset",
        )
        ax.legend(fontsize="small")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial")
    ax.set_title("Spike Raster")
    ax.invert_yaxis()
    return ax


# ---------------------------------------------------------------------------
# New visualisations
# ---------------------------------------------------------------------------


def plot_autocorrelogram(
    spike_times: npt.NDArray,
    trials: npt.NDArray | None = None,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    bin_size: float = 0.001,
    max_lag: float = 0.05,
    refractory_period: float = 0.001,
    normalize: AcgNormalize = "counts",
    ax: plt.Axes | None = None,
    color: str = "steelblue",
) -> plt.Axes:
    """Autocorrelogram bar chart.

    For trial-based recordings, pass *trials* so the underlying
    :func:`autocorrelogram` accumulates pairs within each trial only;
    see that function for the rationale.

    The function emits a :class:`RuntimeWarning` when
    ``refractory_period`` is not a whole multiple of ``bin_size``.
    In that regime the dashed refractory line lands *inside* a bar
    rather than on a bin edge, so visually some pairs in the
    refractory bar are violations and some aren't — the plot can no
    longer be read as "everything left of the line is a violation".
    Pick a ``bin_size`` that divides ``refractory_period`` to silence
    the warning (e.g. ``bin_size=0.0005`` for ``refractory_period=
    0.001``).

    Args:
        spike_times: Spike times in seconds.
        trials: Optional trial index per spike (forwarded to
            :func:`autocorrelogram`).
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to plot.
        bin_size: Bin width (seconds).
        max_lag: Maximum lag (seconds).
        refractory_period: Shown as dashed lines.
        normalize: ``"counts"`` (default) or ``"rate"`` — forwarded to
            :func:`autocorrelogram`.  The y-axis label is set
            accordingly.
        ax: Existing ``Axes``.
        color: Bar colour.

    Returns:
        ``Axes`` with the plot.
    """
    # Warn when the refractory dashed line will not align to a bin
    # edge.  A tiny absolute tolerance protects against float-rounding
    # noise (e.g. ``0.001 - 0.0005 - 0.0005`` is not exactly 0).
    bin_ratio = refractory_period / bin_size
    if abs(bin_ratio - round(bin_ratio)) > 1e-9:
        warnings.warn(
            f"refractory_period={refractory_period} is not a whole "
            f"multiple of bin_size={bin_size} (ratio={bin_ratio:.4f}); "
            "the dashed refractory line will fall inside a bar rather "
            "than on a bin edge, so the plot can no longer be read as "
            "'bars left of the line are violations'. Pick a bin_size "
            "that divides refractory_period to silence this warning.",
            RuntimeWarning,
            stacklevel=2,
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3), dpi=150)
    lags, values = _acg(
        spike_times,
        trials=trials,
        cluster_labels=cluster_labels,
        cluster_id=cluster_id,
        bin_size=bin_size,
        max_lag=max_lag,
        normalize=normalize,
    )
    ax.bar(lags * 1000, values, width=bin_size * 1000 * 0.9, color=color, edgecolor="none")
    ax.axvline(-refractory_period * 1000, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(
        refractory_period * 1000,
        color="red",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label=f"refr. {refractory_period * 1e3:.1f} ms",
    )
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Rate (Hz)" if normalize == "rate" else "Count")
    ax.set_title("Autocorrelogram")
    ax.legend(fontsize=8)
    return ax


def plot_psth(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    bin_size: float = 0.01,
    trial_duration: float = 2.5,
    stim_onset: float | None = None,
    ax: plt.Axes | None = None,
    color: str = "steelblue",
) -> plt.Axes:
    """Peri-stimulus time histogram.

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to plot.
        bin_size: Bin width (seconds).
        trial_duration: Trial duration (seconds).
        stim_onset: Drawn as vertical line if given.
        ax: Existing ``Axes``.
        color: Bar colour.

    Returns:
        ``Axes`` with the PSTH.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), dpi=150)
    centres, rate = _psth(
        spike_times,
        trials,
        cluster_labels,
        cluster_id,
        bin_size=bin_size,
        trial_duration=trial_duration,
    )
    ax.bar(centres, rate, width=bin_size * 0.9, color=color, edgecolor="none")
    if stim_onset is not None:
        ax.axvline(
            stim_onset,
            color="red",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label="Stimulus onset",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("PSTH")
    return ax


def plot_firing_rate_stability(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    window_size: float = 0.5,
    stat: str = "mean",
    trial_duration: float = 2.5,
    ax: plt.Axes | None = None,
    color: str = "steelblue",
) -> plt.Axes:
    """Time-series of a firing-rate statistic across windows.

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to plot.
        window_size: Window duration (seconds).
        stat: Statistic (``"mean"``, ``"cv"``, etc.).
        trial_duration: Trial duration (seconds).
        ax: Existing ``Axes``.
        color: Line colour.

    Returns:
        ``Axes`` with the stability plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3), dpi=150)
    result = _frs(
        spike_times,
        trials,
        cluster_labels,
        cluster_id,
        window_size=window_size,
        stat=stat,
        trial_duration=trial_duration,
    )
    values = result["values"]
    n = len(values)
    edges = np.linspace(0, trial_duration, n + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centres, values, "o-", color=color, linewidth=1.5, markersize=4)
    ax.axhline(
        result["mean"],
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"mean = {result['mean']:.3f}",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(stat.upper())
    ax.set_title(f"Firing rate stability ({stat})")
    ax.legend(fontsize=8)
    return ax


def plot_first_spike_latency(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    stim_onset: float = 0.5,
    n_bins: int = 30,
    ax: plt.Axes | None = None,
    color: str = "steelblue",
) -> plt.Axes:
    """Histogram of first-spike latencies.

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to plot.
        stim_onset: Stimulus onset (seconds).
        n_bins: Number of histogram bins.
        ax: Existing ``Axes``.
        color: Bar colour.

    Returns:
        ``Axes`` with the latency histogram.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=150)
    result = _fsl(spike_times, trials, cluster_labels, cluster_id, stim_onset=stim_onset)
    lats = result["latencies"]
    valid = lats[~np.isnan(lats)]
    if len(valid) > 0:
        ax.hist(valid * 1000, bins=n_bins, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axvline(
            result["mean"] * 1000,
            color="tomato",
            linestyle="--",
            linewidth=1.2,
            label=f"mean = {result['mean'] * 1000:.1f} ms",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"First spike latency (responsive: {result['frac_responsive']:.0%})")
    return ax


def plot_trial_reliability_heatmap(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    bin_size: float = 0.01,
    trial_duration: float = 2.5,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of per-trial PSTHs (rows = trials, columns = time bins).

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to plot.
        bin_size: Bin width (seconds).
        trial_duration: Trial duration (seconds).
        ax: Existing ``Axes``.

    Returns:
        ``Axes`` with the heatmap.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), dpi=150)

    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        trials = trials[mask]

    # ``np.arange`` with a float step is fragile under rounding: the
    # final edge can land slightly above or below ``trial_duration``,
    # producing an over- or under-sized last bin and an extent that no
    # longer matches the data.  Use the same ``ceil + linspace`` recipe
    # as ``psth()`` and ``firing_rate_stability()``.
    n_bins = int(np.ceil(trial_duration / bin_size))
    edges = np.linspace(0.0, trial_duration, n_bins + 1)
    unique_trials = np.sort(np.unique(trials))
    mat = np.zeros((len(unique_trials), len(edges) - 1))
    for i, t in enumerate(unique_trials):
        t_spikes = spike_times[trials == t]
        mat[i], _ = np.histogram(t_spikes, bins=edges)

    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="hot",
        extent=[0, trial_duration, len(unique_trials), 0],
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial")
    ax.set_title("Trial-by-trial spike counts")
    plt.colorbar(im, ax=ax, label="Spike count")
    return ax
