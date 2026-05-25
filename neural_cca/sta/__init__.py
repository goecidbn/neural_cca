"""Spike train analysis for extracellular recordings."""

from .analysis import (
    minimal_spike_train_analysis,
    calc_mfr_trial,
    isi_violation_rate,
    firing_rate_stability,
    autocorrelogram,
    fano_factor,
    local_variation,
    cv_log_isi,
    psth,
    trial_to_trial_reliability,
    trial_to_trial_correlation_matrix,
    first_spike_latency,
)
from .plotting import (
    plot_isi_histogram,
    plot_waveform_snippets,
    plot_spike_raster,
    plot_autocorrelogram,
    plot_psth,
    plot_firing_rate_stability,
    plot_first_spike_latency,
    plot_trial_reliability_heatmap,
)

__all__ = [
    "minimal_spike_train_analysis",
    "calc_mfr_trial",
    "isi_violation_rate",
    "firing_rate_stability",
    "autocorrelogram",
    "fano_factor",
    "local_variation",
    "cv_log_isi",
    "psth",
    "trial_to_trial_reliability",
    "trial_to_trial_correlation_matrix",
    "first_spike_latency",
    "plot_isi_histogram",
    "plot_waveform_snippets",
    "plot_spike_raster",
    "plot_autocorrelogram",
    "plot_psth",
    "plot_firing_rate_stability",
    "plot_first_spike_latency",
    "plot_trial_reliability_heatmap",
]
