"""Unified spike sorting, spike-train statistics, and tuning analysis.

Subpackages:
    sorting: Spike clustering, quality metrics, batch processing.
    spike_train: Spike-train statistics (MFR, CV, LvR, PSTH, ACG, etc.).
        Previously called ``sta``; the old import path is a
        deprecation shim retained for backwards compatibility.
    tuning: Orientation selectivity and tuning curves.

Convenience re-exports allow both::

    from neural_cca import get_os_metrics
    from neural_cca.tuning import get_os_metrics
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("neural-cca")
except PackageNotFoundError:
    # Fallback for editable installs / development without pip install:
    # read the version from pyproject.toml so it stays in sync.
    try:
        try:
            import tomllib as _tomllib  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as _tomllib  # Python 3.10 backport
        from pathlib import Path as _Path

        _pyproject = _Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(_pyproject, "rb") as _f:
            __version__ = _tomllib.load(_f)["project"]["version"]
    except Exception:
        __version__ = "0.0.0"

# Build __all__ from the names actually imported above.
# This avoids duplicating the list and stays in sync automatically.
import types as _types

from ._utils import (
    circ_dist,
    circ_mean,
    guarded_divide,
    make_rng,
    wrap180,
    wrap360,
)
from .sorting import (
    SortingData,
    SortingResult,
    amplitude_drift,
    # Batch driver
    batch_sort_experiment,
    calc_weighted_snr,
    contamination_rate_hill,
    d_prime,
    d_prime_pairwise_matrix,
    est_snr,
    evaluate_os_per_cluster,
    evaluate_sorting,
    find_optimal_k,
    fraction_missing,
    # Sorting metrics
    isolation_distance,
    l_ratio,
    load_from_arrays,
    neg_silhouette_score,
    peak_amplitude_snr,
    plot_amplitude_drift,
    plot_amplitude_histogram,
    plot_d_prime_matrix,
    plot_k_search,
    # Sorting plots
    plot_metric_bars,
    plot_sorting_summary,
    plot_waveform_stability,
    read_zarr_sorting,
    rpvs,
    run_sorting_pipeline,
    sort_spikes,
    spikes_before_stimulus,
    to_zarr_clustered,
    # Zarr export/import
    to_zarr_flat,
    waveform_stability,
)
from .spike_train import (
    autocorrelogram,
    calc_mfr_trial,
    cv_log_isi,
    fano_factor,
    firing_rate_stability,
    first_spike_latency,
    first_spike_latency_thresholded,
    # Spike train analyses
    isi_violation_rate,
    local_variation,
    minimal_spike_train_analysis,
    # Spike train plots
    plot_autocorrelogram,
    plot_firing_rate_stability,
    plot_first_spike_latency,
    plot_isi_histogram,
    plot_psth,
    plot_spike_raster,
    plot_trial_reliability_heatmap,
    plot_waveform_snippets,
    psth,
    trial_to_trial_correlation_matrix,
    trial_to_trial_reliability,
)
from .synthetic import (
    TwoUnitDemo,
    make_tuned_spikes,
    make_two_unit_demo,
    poisson_train,
)
from .tuning import (
    OsMetricsResult,
    anova_across_orientations,
    bootstrap_ci,
    bootstrap_ci_strata,
    circular_variance,
    compute_f0_f1_f2,
    cross_orientation_suppression,
    # selectivity
    dosi_circular_normalised,
    double_gaussian_fit,
    dsi_two_point,
    f1_phase,
    gdsi,
    get_os_metrics,
    goodness_of_fit,
    gosi,
    # modulation
    modulation_ratio_per_orientation,
    noise_correlations,
    # population
    orientation_map_statistics,
    # tuning plots
    orientation_scatter_vm,
    # statistics
    orientation_selectivity_significance,
    osi_two_point,
    plot_direction_polar,
    plot_f1_phase,
    plot_f1f0_bars,
    plot_modulation_ratio,
    plot_noise_correlation_matrix,
    plot_population_orientation_histogram,
    plot_psth_by_orientation,
    plot_signal_correlation_matrix,
    plot_temporal_frequency_tuning,
    plot_tuning_curve,
    preferred_dori,
    signal_correlations,
    # utils
    steps2degree,
    # temporal
    temporal_frequency_tuning,
    # tuning
    tuning_bandwidth,
    tuning_curve_interpolation,
    # fitting
    von_mises_fit,
)

__all__ = sorted(
    name
    for name, obj in vars().items()
    if not name.startswith("_")
    and not isinstance(obj, _types.ModuleType)
    and name not in ("__version__", "PackageNotFoundError")
) + ["__version__"]
