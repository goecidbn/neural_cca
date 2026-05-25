"""Sorting diagnostic visualisation."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

# Cross-package imports for the optional polar-MFR panel.  These are
# part of the same distribution and have no back-edge to sorting,
# so importing them at module top is safe and replaces the previous
# in-function ``try/except ImportError``.
from ..spike_train.analysis import calc_mfr_trial
from ..tuning.plotting import orientation_scatter_vm
from .containers import SortingData
from .metrics import d_prime_pairwise_matrix

__all__ = [
    "plot_sorting_summary",
    "plot_k_search",
    "plot_metric_bars",
    "plot_d_prime_matrix",
    "plot_waveform_stability",
    "plot_amplitude_drift",
    "plot_amplitude_histogram",
]

_COLORS_LIGHT = [
    "cornflowerblue",
    "salmon",
    "lightgreen",
    "plum",
    "sandybrown",
    "lightskyblue",
    "khaki",
    "thistle",
]
_COLORS_DARK = [
    "blue",
    "red",
    "green",
    "purple",
    "darkorange",
    "darkblue",
    "olive",
    "darkviolet",
]


def plot_sorting_summary(
    data: SortingData,
    cluster_labels: npt.NDArray[np.int64],
    invert_waveforms: bool = True,
    hist_bins: int = 250,
    dpi: int = 300,
    figsize_per_cluster: tuple[float, float] = (9.0, 2.5),
) -> plt.Figure:
    """Three-panel diagnostic figure for each cluster.

    Columns:
        A — Overlay of all waveforms + mean (dark line).
        B — Spike raster + PSTH histogram.
        C — Polar scatter of trial-wise mean firing rate (if
        tuning is available and angles are present).

    Args:
        data: ``SortingData`` container.
        cluster_labels: Cluster labels per spike.
        invert_waveforms: If ``True``, plot ``-waveform``.
        hist_bins: Number of histogram bins for the PSTH.
        dpi: Figure resolution.
        figsize_per_cluster: ``(width, height)`` per cluster row.

    Returns:
        ``matplotlib.figure.Figure``.
    """
    unique_labels = np.sort(np.unique(cluster_labels))
    n_cl = len(unique_labels)

    w, h = figsize_per_cluster
    fig = plt.figure(figsize=(w, h * n_cl), dpi=dpi, layout="constrained")
    subfigs = fig.subfigures(nrows=n_cl, ncols=1)
    if n_cl == 1:
        subfigs = [subfigs]

    t_ms = data.time_axis_ms
    sign = -1.0 if invert_waveforms else 1.0
    has_angles = data.angles is not None and len(data.angles) > 0
    mosaic = "ABC" if has_angles else "AB"

    for row, cl in enumerate(unique_labels):
        cl_mask = cluster_labels == cl
        c_light = _COLORS_LIGHT[row % len(_COLORS_LIGHT)]
        c_dark = _COLORS_DARK[row % len(_COLORS_DARK)]

        per_kw = {"C": {"projection": "polar"}} if has_angles else {}
        axs = subfigs[row].subplot_mosaic(mosaic, per_subplot_kw=per_kw)

        # --- A: Waveforms ---
        wv_cl = data.waveforms[cl_mask]
        axs["A"].set_title(f"Cluster {cl} waveforms (n={len(wv_cl)})")
        axs["A"].set_ylabel("Amplitude (a.u.)")
        axs["A"].set_xlabel("Time (ms)")
        axs["A"].plot(
            t_ms, sign * wv_cl.T, color=c_light, alpha=0.15, linewidth=0.3, rasterized=True
        )
        axs["A"].plot(t_ms, sign * wv_cl.mean(axis=0), color=c_dark, linewidth=2)

        # --- B: Spike raster + PSTH ---
        st_cl = data.spike_times[cl_mask]
        axs["B"].set_title(f"Cluster {cl} spike histogram")
        axs["B"].axvline(
            x=data.stim_window[0],
            color="black",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
            label="Stimulus onset",
        )
        axs["B"].scatter(
            st_cl, np.zeros_like(st_cl), marker="|", linewidth=0.1, color="black", s=10
        )
        axs["B"].hist(st_cl, bins=hist_bins, color=c_light, alpha=0.7)
        axs["B"].set_xlim(0, data.stim_window[1])
        axs["B"].set_ylabel("Count")
        axs["B"].set_xlabel("Time (s)")

        # --- C: Polar MFR (optional) ---
        if has_angles:
            mfrs = calc_mfr_trial(
                data.spike_times,
                data.trials,
                all_clusters=False,
                cluster_labels=cluster_labels,
                cluster_id=int(cl),
                stim_window=data.stim_window,
                n_trials=data.n_trials,
            )
            orientation_scatter_vm(
                response=list(mfrs.values()),
                orientations=data.angles,
                ax=axs["C"],
                color=c_light,
            )

    return fig


def plot_k_search(
    scores: dict[int, float],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot silhouette scores vs. number of clusters.

    Args:
        scores: ``{k: silhouette_score}`` as returned by
            ``find_optimal_k``.
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        ``Axes`` with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ks = sorted(scores)
    vals = [scores[k] for k in ks]
    ax.plot(ks, vals, "o-", color="steelblue")
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    ax.axvline(best, color="tomato", linestyle="--", label=f"best k={best}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette score")
    ax.set_title("Cluster-number selection")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# New metric visualisations
# ---------------------------------------------------------------------------


def plot_metric_bars(
    metric_dict: dict[int, float],
    title: str = "Metric per cluster",
    ylabel: str = "Value",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of a per-cluster metric.

    Args:
        metric_dict: ``{cluster_id: value}`` as returned by e.g.
            ``isolation_distance(..., all_clusters=True)``.
        title: Plot title.
        ylabel: Y-axis label.
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        ``Axes`` with the bar chart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=150)
    cids = sorted(metric_dict)
    vals = [metric_dict[c] for c in cids]
    colours = [_COLORS_LIGHT[i % len(_COLORS_LIGHT)] for i in range(len(cids))]
    ax.bar([str(c) for c in cids], vals, color=colours, edgecolor="gray")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def plot_d_prime_matrix(
    X: npt.NDArray,
    cluster_labels: npt.NDArray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of pairwise d-prime between clusters.

    Uses the canonical :func:`d_prime_pairwise_matrix` from
    ``sorting.metrics`` so the displayed values match what
    :func:`d_prime` computes (both consume the same helper).

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        cluster_labels: Cluster label per sample.
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        ``Axes`` with the heatmap.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4), dpi=150)

    mat, unique = d_prime_pairwise_matrix(X, np.asarray(cluster_labels))
    n = len(unique)

    im = ax.imshow(mat, cmap="viridis", aspect="equal")
    ax.set_xticks(range(n), [str(int(c)) for c in unique])
    ax.set_yticks(range(n), [str(int(c)) for c in unique])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    ax.set_title("Pairwise d-prime")
    plt.colorbar(im, ax=ax, label="d'")

    # Annotate cells
    finite = mat[np.isfinite(mat)]
    midpoint = float(finite.mean()) if finite.size else 0.0
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > midpoint else "black",
            )
    return ax


def plot_waveform_stability(
    spike_times: npt.NDArray,
    wv: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    percentiles: Sequence[float] = (25, 75),
    waveform_fs: float = 32_000.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay mean waveforms from early and late time windows.

    Args:
        spike_times: Spike times (for temporal ordering).
        wv: Waveform matrix ``(n_spikes, snippet_length)``.
        cluster_labels: Cluster labels (optional).
        cluster_id: If given with *cluster_labels*, restrict to this
            cluster.
        percentiles: Time percentiles to split at.
        waveform_fs: Sampling rate for x-axis in ms.
        ax: Existing ``Axes``.

    Returns:
        ``Axes`` with overlaid early/late mean waveforms.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=150)

    if cluster_labels is not None and cluster_id is not None:
        mask = np.asarray(cluster_labels) == cluster_id
        spike_times = spike_times[mask]
        wv = wv[mask]

    if len(spike_times) < 4:
        ax.set_title("Insufficient spikes")
        return ax

    order = np.argsort(spike_times)
    wv_sorted = wv[order]
    t_sorted = spike_times[order]
    t_ms = np.arange(wv.shape[1]) / waveform_fs * 1000.0

    pct_vals = np.percentile(t_sorted, percentiles)
    # Early window: before first percentile
    early_mask = t_sorted < pct_vals[0]
    # Late window: after last percentile
    late_mask = t_sorted >= pct_vals[-1]

    if early_mask.sum() > 0:
        ax.plot(
            t_ms,
            np.mean(wv_sorted[early_mask], axis=0),
            color="steelblue",
            linewidth=2,
            label=f"Early (<{percentiles[0]}%)",
        )
    if late_mask.sum() > 0:
        ax.plot(
            t_ms,
            np.mean(wv_sorted[late_mask], axis=0),
            color="tomato",
            linewidth=2,
            label=f"Late (>{percentiles[-1]}%)",
        )

    if early_mask.sum() > 0 and late_mask.sum() > 0:
        r, _ = sp_stats.pearsonr(
            np.mean(wv_sorted[early_mask], axis=0),
            np.mean(wv_sorted[late_mask], axis=0),
        )
        ax.set_title(f"Waveform stability (r = {r:.3f})")
    else:
        ax.set_title("Waveform stability")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.legend(fontsize=8)
    return ax


def plot_amplitude_drift(
    wv: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter of peak amplitude vs. spike index with trend line.

    Args:
        wv: Waveform matrix ``(n_spikes, snippet_length)``.
        cluster_labels: Cluster labels (optional).
        cluster_id: If given with *cluster_labels*, restrict to this
            cluster.
        ax: Existing ``Axes``.

    Returns:
        ``Axes`` with the scatter plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3), dpi=150)

    if cluster_labels is not None and cluster_id is not None:
        wv = wv[np.asarray(cluster_labels) == cluster_id]

    amps = np.max(wv, axis=1) - np.min(wv, axis=1)
    idx = np.arange(len(amps))

    ax.scatter(idx, amps, s=1, alpha=0.3, color="steelblue", rasterized=True)

    if len(amps) >= 3:
        r, p = sp_stats.spearmanr(idx, amps)
        # Linear trend for visualisation
        slope, intercept = np.polyfit(idx, amps, 1)
        ax.plot(
            idx,
            slope * idx + intercept,
            color="tomato",
            linewidth=1.5,
            label=f"Spearman r={r:.3f}, p={p:.2e}",
        )
        ax.legend(fontsize=8)

    ax.set_xlabel("Spike index (chronological)")
    ax.set_ylabel("Peak-to-peak amplitude")
    ax.set_title("Amplitude drift")
    return ax


def plot_amplitude_histogram(
    wv: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    n_bins: int = 50,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of peak amplitudes with Gaussian fit.

    Shaded region shows estimated missing spikes below the detection
    threshold.

    Args:
        wv: Waveform matrix ``(n_spikes, snippet_length)``.
        cluster_labels: Cluster labels (optional).
        cluster_id: If given with *cluster_labels*, restrict to this
            cluster.
        n_bins: Number of histogram bins.
        ax: Existing ``Axes``.

    Returns:
        ``Axes`` with the histogram.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=150)

    if cluster_labels is not None and cluster_id is not None:
        wv = wv[np.asarray(cluster_labels) == cluster_id]

    amps = np.max(wv, axis=1) - np.min(wv, axis=1)
    ax.hist(
        amps,
        bins=n_bins,
        color="steelblue",
        alpha=0.7,
        density=True,
        edgecolor="white",
        linewidth=0.3,
    )

    if len(amps) >= 10:
        mu, sigma = sp_stats.norm.fit(amps)
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(
            x, sp_stats.norm.pdf(x, mu, sigma), color="tomato", linewidth=1.5, label="Gaussian fit"
        )
        threshold = amps.min()
        ax.axvline(
            threshold, color="gray", linestyle="--", linewidth=1, label=f"Min amp = {threshold:.1f}"
        )
        # Shade missing region
        x_fill = np.linspace(mu - 4 * sigma, threshold, 100)
        ax.fill_between(
            x_fill,
            sp_stats.norm.pdf(x_fill, mu, sigma),
            color="tomato",
            alpha=0.2,
            label="Est. missing",
        )
        frac = sp_stats.norm.cdf(threshold, loc=mu, scale=sigma)
        ax.set_title(f"Amplitude distribution (est. missing: {frac:.1%})")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Amplitude distribution")

    ax.set_xlabel("Peak-to-peak amplitude")
    ax.set_ylabel("Density")
    return ax
