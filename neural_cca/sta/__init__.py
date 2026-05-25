"""Spike train analysis for extracellular recordings."""

from .analysis import (
    autocorrelogram,
    calc_mfr_trial,
    cv_log_isi,
    fano_factor,
    firing_rate_stability,
    first_spike_latency,
    isi_violation_rate,
    local_variation,
    minimal_spike_train_analysis,
    psth,
    trial_to_trial_correlation_matrix,
    trial_to_trial_reliability,
)
from .plotting import (
    plot_autocorrelogram,
    plot_firing_rate_stability,
    plot_first_spike_latency,
    plot_isi_histogram,
    plot_psth,
    plot_spike_raster,
    plot_trial_reliability_heatmap,
    plot_waveform_snippets,
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
