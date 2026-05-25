"""Orientation selectivity and tuning analysis visualisation.

Provides polar scatter, tuning curves, direction polar plots,
F1/F0 bars, PSTH-by-orientation heatmaps, phase plots,
population histograms, correlation matrices, modulation ratio
bars, and temporal frequency tuning plots.

All functions follow the ``ax=None`` pattern: when *ax* is ``None``
a new figure is created; otherwise the plot is drawn on the
provided ``Axes``.  Every function returns the ``Axes`` object.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

__all__ = [
    "orientation_scatter_vm",
    "plot_tuning_curve",
    "plot_direction_polar",
    "plot_f1f0_bars",
    "plot_psth_by_orientation",
    "plot_f1_phase",
    "plot_population_orientation_histogram",
    "plot_noise_correlation_matrix",
    "plot_signal_correlation_matrix",
    "plot_modulation_ratio",
    "plot_temporal_frequency_tuning",
]


# ---------------------------------------------------------------------------
# Original v0.1 plot
# ---------------------------------------------------------------------------


def orientation_scatter_vm(
    response: npt.NDArray[np.float64],
    orientations: npt.NDArray[np.float64] | int,
    ax: plt.Axes | None = None,
    color: str = "blue",
) -> plt.Axes:
    """Polar scatter plot of firing rates by orientation.

    Args:
        response: Firing rates (one per orientation entry).
        orientations: Angles in degrees, or an int *N* to generate
            *N* equally-spaced angles in [0, 360).
        ax: Existing polar ``Axes`` (created if ``None``).
        color: Marker colour.

    Returns:
        The polar ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=300, subplot_kw={"projection": "polar"})

    if isinstance(orientations, (int, np.integer)):
        orientations = np.linspace(0, 360, int(orientations), endpoint=False)
    orientations = np.asarray(orientations, dtype=np.float64)

    ax.scatter(np.deg2rad(orientations), response, color=color, alpha=0.5)
    ax.set_rlabel_position(0)
    return ax


# ---------------------------------------------------------------------------
# v0.3.0 plots
# ---------------------------------------------------------------------------


def plot_tuning_curve(
    responses: npt.ArrayLike,
    orientations: npt.ArrayLike,
    fitted: npt.ArrayLike | None = None,
    sem: npt.ArrayLike | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Mean ± SEM tuning curve with optional fitted-model overlay.

    Args:
        responses: Mean firing rates at each orientation.
        orientations: Stimulus orientations in degrees.
        fitted: Fitted curve values at each orientation (optional).
        sem: Standard error of the mean at each orientation (optional).
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        The ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), dpi=150)

    orientations = np.asarray(orientations, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)

    if sem is not None:
        sem = np.asarray(sem, dtype=np.float64)
        ax.fill_between(
            orientations,
            responses - sem,
            responses + sem,
            alpha=0.2,
            color="steelblue",
        )

    ax.plot(orientations, responses, "o-", color="steelblue", label="Data")

    if fitted is not None:
        fitted = np.asarray(fitted, dtype=np.float64)
        ax.plot(orientations, fitted, "--", color="crimson", label="Fit")
        ax.legend()

    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Tuning Curve")
    return ax


def plot_direction_polar(
    response: npt.ArrayLike,
    orientations: npt.ArrayLike,
    show_vector: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Polar plot of direction tuning with optional DSI vector arrow.

    Args:
        response: Mean firing rates at each direction.
        orientations: Stimulus directions in degrees.
        show_vector: If ``True``, draw the DSI vector arrow.
        ax: Existing polar ``Axes`` (created if ``None``).

    Returns:
        The polar ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=150, subplot_kw={"projection": "polar"})

    response = np.asarray(response, dtype=np.float64)
    orientations = np.asarray(orientations, dtype=np.float64)
    theta = np.deg2rad(orientations)

    # Close the curve
    theta_closed = np.append(theta, theta[0])
    resp_closed = np.append(response, response[0])

    ax.plot(theta_closed, resp_closed, "o-", color="steelblue")
    ax.fill(theta_closed, resp_closed, alpha=0.15, color="steelblue")

    if show_vector:
        vec = np.sum(response * np.exp(1j * theta))
        vec_angle = np.angle(vec)
        vec_len = np.abs(vec) / np.sum(response) * np.max(response)
        ax.annotate(
            "",
            xy=(vec_angle, vec_len),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2),
        )

    ax.set_title("Direction Tuning", pad=15)
    return ax


def plot_f1f0_bars(
    f1f0_ratios: npt.ArrayLike,
    cluster_ids: npt.ArrayLike | None = None,
    threshold: float = 1.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of per-cluster F1/F0 ratios with simple/complex line.

    Args:
        f1f0_ratios: F1/F0 ratio per cluster.
        cluster_ids: Labels for each bar (defaults to 0, 1, 2, …).
        threshold: F1/F0 threshold distinguishing simple (above) from
            complex (below) cells.
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        The ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), dpi=150)

    f1f0 = np.asarray(f1f0_ratios, dtype=np.float64)
    n = len(f1f0)
    if cluster_ids is None:
        cluster_ids = np.arange(n)
    cluster_ids = np.asarray(cluster_ids)

    colours = ["steelblue" if v >= threshold else "coral" for v in f1f0]
    ax.bar(range(n), f1f0, color=colours, edgecolor="black", linewidth=0.5)
    ax.axhline(threshold, color="grey", linestyle="--", linewidth=1, label=f"Threshold={threshold}")
    ax.set_xticks(range(n))
    ax.set_xticklabels(cluster_ids)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("F1/F0")
    ax.set_title("Modulation Ratio (F1/F0)")
    ax.legend()
    return ax


def plot_psth_by_orientation(
    psth_dict: dict[float, npt.NDArray],
    time_axis: npt.ArrayLike | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of PSTHs arranged by orientation.

    Args:
        psth_dict: Mapping ``{angle_deg: psth_array}``.
        time_axis: Time bin centres (seconds).  Inferred if ``None``.
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        The ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), dpi=150)

    sorted_angles = sorted(psth_dict.keys())
    matrix = np.array([psth_dict[a] for a in sorted_angles])

    if time_axis is None:
        time_axis = np.arange(matrix.shape[1])
    time_axis = np.asarray(time_axis, dtype=np.float64)

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], 0, len(sorted_angles)],
        cmap="viridis",
    )
    ax.set_yticks(np.arange(len(sorted_angles)) + 0.5)
    ax.set_yticklabels([f"{a:.0f}°" for a in sorted_angles])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Orientation")
    ax.set_title("PSTH by Orientation")
    plt.colorbar(im, ax=ax, label="Rate (Hz)")
    return ax


def plot_f1_phase(
    phases: npt.ArrayLike,
    orientations: npt.ArrayLike,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of F1 phase vs. orientation.

    Args:
        phases: F1 phase (radians) at each orientation.
        orientations: Stimulus orientations in degrees.
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        The ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), dpi=150)

    phases = np.asarray(phases, dtype=np.float64)
    orientations = np.asarray(orientations, dtype=np.float64)

    ax.scatter(orientations, np.rad2deg(phases), color="steelblue", edgecolors="black", s=50)
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("F1 Phase (°)")
    ax.set_title("F1 Phase vs. Orientation")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    return ax


def plot_population_orientation_histogram(
    pref_oris: npt.ArrayLike,
    n_bins: int = 18,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Circular histogram of preferred orientations across a population.

    Bins are **centred on** multiples of ``180 / n_bins`` rather than
    edged on them.  For an experiment that samples orientations on a
    discrete grid (e.g. 0°, 30°, 60°, …) and ``n_bins`` matching that
    grid, every sampled orientation lands at the centre of its own bin
    — so a cell whose preferred orientation is 90° produces a bar at
    the "90°" tick instead of half a bin past it.  An earlier version
    placed bin edges at multiples of ``bin_width``, which put a value
    of 90° into the half-open bin ``[90°, 120°)`` and drew the bar at
    105°: visually wrong by half a bin in the matplotlib polar default
    orientation.

    Args:
        pref_oris: Preferred orientations in degrees (one per neuron).
        n_bins: Number of angular bins (spanning 0-180°).  Pass the
            number of distinct sampled orientations for the cleanest
            display.
        ax: Existing polar ``Axes`` (created if ``None``).

    Returns:
        The polar ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=150, subplot_kw={"projection": "polar"})

    pref_oris = np.asarray(pref_oris, dtype=np.float64)

    # Shift the data by half a bin width before histogramming, so the
    # bins of the standard ``[0, 180]`` partition become *centred* on
    # ``0, bw, 2*bw, …, 180 - bw`` instead of edged on them.  Equivalent
    # to building bin edges at ``[-bw/2, bw/2, 3*bw/2, …]`` and wrapping
    # the [-bw/2, 0) sliver back to the [180-bw/2, 180) end.
    bin_width = 180.0 / n_bins
    half = bin_width / 2.0
    shifted = (pref_oris % 180.0 + half) % 180.0
    bin_edges = np.linspace(0.0, 180.0, n_bins + 1)
    counts, _ = np.histogram(shifted, bins=bin_edges)

    # Bar centres on the orientation grid (0, bw, 2*bw, …, 180 - bw)
    # and the same set mirrored to the lower half-circle.
    centres_deg = np.arange(n_bins) * bin_width
    theta = np.deg2rad(centres_deg)
    theta_full = np.concatenate([theta, theta + np.pi])
    counts_full = np.concatenate([counts, counts])
    width = np.deg2rad(bin_width)

    ax.bar(
        theta_full,
        counts_full,
        width=width,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )

    # Tick layout: every 30° across the full circle.  Independent of
    # ``n_bins`` so the labels stay legible for fine binning, and the
    # canonical sampled orientations (0, 30, 60, 90, 120, 150) always
    # appear as labelled ticks.
    tick_angles_deg = np.arange(0, 360, 30)
    tick_labels = [f"{int(a % 180)}°" for a in tick_angles_deg]
    ax.set_xticks(np.deg2rad(tick_angles_deg))
    ax.set_xticklabels(tick_labels)

    ax.set_title("Preferred Orientation Distribution", pad=15)
    return ax


def plot_noise_correlation_matrix(
    corr_matrix: npt.ArrayLike,
    ids: npt.ArrayLike | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of noise correlations (RdBu_r colourmap, diagonal masked).

    Args:
        corr_matrix: ``(n, n)`` noise correlation matrix.
        ids: Neuron labels for axes.
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        The ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=150)

    corr = np.asarray(corr_matrix, dtype=np.float64).copy()
    np.fill_diagonal(corr, np.nan)
    n = corr.shape[0]

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    if ids is not None:
        ids = np.asarray(ids)
        ax.set_xticks(range(n))
        ax.set_xticklabels(ids, rotation=45)
        ax.set_yticks(range(n))
        ax.set_yticklabels(ids)
    ax.set_title("Noise Correlations")
    plt.colorbar(im, ax=ax, label="r")
    return ax


def plot_signal_correlation_matrix(
    corr_matrix: npt.ArrayLike,
    ids: npt.ArrayLike | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of signal correlations (viridis colourmap).

    Args:
        corr_matrix: ``(n, n)`` signal correlation matrix.
        ids: Neuron labels for axes.
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        The ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=150)

    corr = np.asarray(corr_matrix, dtype=np.float64).copy()
    np.fill_diagonal(corr, np.nan)
    n = corr.shape[0]

    im = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1, aspect="equal")
    if ids is not None:
        ids = np.asarray(ids)
        ax.set_xticks(range(n))
        ax.set_xticklabels(ids, rotation=45)
        ax.set_yticks(range(n))
        ax.set_yticklabels(ids)
    ax.set_title("Signal Correlations")
    plt.colorbar(im, ax=ax, label="r")
    return ax


def plot_modulation_ratio(
    mod_dict: dict[float, float],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Per-orientation F1/F0 modulation ratio bar chart.

    Args:
        mod_dict: Mapping ``{angle_deg: f1_f0_ratio}`` as returned by
            :func:`~tuning.modulation.modulation_ratio_per_orientation`.
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        The ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4), dpi=150)

    angles = sorted(mod_dict.keys())
    ratios = [mod_dict[a] for a in angles]

    ax.bar(range(len(angles)), ratios, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a:.0f}°" for a in angles])
    ax.set_xlabel("Orientation (°)")
    ax.set_ylabel("F1/F0")
    ax.set_title("Modulation Ratio per Orientation")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    return ax


def plot_temporal_frequency_tuning(
    amplitudes: npt.ArrayLike,
    temporal_freqs: npt.ArrayLike,
    fit_curve: npt.ArrayLike | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Response amplitude vs. temporal frequency with optional fit.

    Args:
        amplitudes: Response amplitude at each TF.
        temporal_freqs: Tested temporal frequencies (Hz).
        fit_curve: Fitted curve values at each TF (optional).
        ax: Existing ``Axes`` (created if ``None``).

    Returns:
        The ``Axes``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), dpi=150)

    tfs = np.asarray(temporal_freqs, dtype=np.float64)
    amps = np.asarray(amplitudes, dtype=np.float64)

    ax.semilogx(tfs, amps, "o-", color="steelblue", label="Data", base=2)

    if fit_curve is not None:
        fit_curve = np.asarray(fit_curve, dtype=np.float64)
        ax.semilogx(tfs, fit_curve, "--", color="crimson", label="Fit", base=2)
        ax.legend()

    ax.set_xlabel("Temporal Frequency (Hz)")
    ax.set_ylabel("Response Amplitude")
    ax.set_title("Temporal Frequency Tuning")
    return ax
