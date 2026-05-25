"""Spike sorting pipeline and quality metrics."""

from .containers import SortingData, SortingResult
from .io_util import load_from_arrays
from .sorting import (
    find_optimal_k,
    sort_spikes,
    evaluate_sorting,
    evaluate_os_per_cluster,
    run_sorting_pipeline,
)
from .metrics import (
    neg_silhouette_score,
    spikes_before_stimulus,
    est_snr,
    calc_weighted_snr,
    rpvs,
    isolation_distance,
    l_ratio,
    d_prime,
    d_prime_pairwise_matrix,
    peak_amplitude_snr,
    waveform_stability,
    amplitude_drift,
    fraction_missing,
)
from .plotting import (
    plot_sorting_summary,
    plot_k_search,
    plot_metric_bars,
    plot_d_prime_matrix,
    plot_waveform_stability,
    plot_amplitude_drift,
    plot_amplitude_histogram,
)
from .io_util import to_zarr_flat, to_zarr_clustered, read_zarr_sorting

__all__ = [
    "SortingData",
    "load_from_arrays",
    "SortingResult",
    "find_optimal_k",
    "sort_spikes",
    "evaluate_sorting",
    "evaluate_os_per_cluster",
    "run_sorting_pipeline",
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
