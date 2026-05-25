"""Spike sorting pipeline and quality metrics."""

from .batch import batch_sort_experiment
from .containers import SortingData, SortingResult
from .io_util import load_from_arrays, read_zarr_sorting, to_zarr_clustered, to_zarr_flat
from .metrics import (
    amplitude_drift,
    calc_weighted_snr,
    d_prime,
    d_prime_pairwise_matrix,
    est_snr,
    fraction_missing,
    isolation_distance,
    l_ratio,
    neg_silhouette_score,
    peak_amplitude_snr,
    rpvs,
    spikes_before_stimulus,
    waveform_stability,
)
from .plotting import (
    plot_amplitude_drift,
    plot_amplitude_histogram,
    plot_d_prime_matrix,
    plot_k_search,
    plot_metric_bars,
    plot_sorting_summary,
    plot_waveform_stability,
)
from .sorting import (
    evaluate_os_per_cluster,
    evaluate_sorting,
    find_optimal_k,
    run_sorting_pipeline,
    sort_spikes,
)

__all__ = [
    "SortingData",
    "load_from_arrays",
    "SortingResult",
    "find_optimal_k",
    "sort_spikes",
    "evaluate_sorting",
    "evaluate_os_per_cluster",
    "run_sorting_pipeline",
    "batch_sort_experiment",
    "neg_silhouette_score",
    "spikes_before_stimulus",
    "est_snr",
    "calc_weighted_snr",
    "rpvs",
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "d_prime_pairwise_matrix",
    "peak_amplitude_snr",
    "waveform_stability",
    "amplitude_drift",
    "fraction_missing",
    "plot_sorting_summary",
    "plot_k_search",
    "plot_metric_bars",
    "plot_d_prime_matrix",
    "plot_waveform_stability",
    "plot_amplitude_drift",
    "plot_amplitude_histogram",
    "to_zarr_flat",
    "to_zarr_clustered",
    "read_zarr_sorting",
]
