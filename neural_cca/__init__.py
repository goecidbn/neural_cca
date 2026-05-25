"""Unified spike sorting, STA, and tuning analysis for extracellular recordings.

Subpackages:
    sorting: Spike clustering, quality metrics, batch processing.
    sta: Spike train statistics (MFR, CV, LvR, PSTH, ACG, etc.).
    tuning: Orientation selectivity and tuning curves.

Convenience re-exports allow both::

    from neural_cca import get_os_metrics
    from neural_cca.tuning import get_os_metrics
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("neural-cca")
except PackageNotFoundError:
    # Fallback for editable installs / development without pip install:
    # read the version from pyproject.toml so it stays in sync.
    try:
        from pathlib import Path as _Path
        import tomllib as _tomllib

        _pyproject = _Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(_pyproject, "rb") as _f:
            __version__ = _tomllib.load(_f)["project"]["version"]
    except Exception:
        __version__ = "0.0.0"

from .sorting import (
    SortingData,
    load_from_arrays,
    SortingResult,
    find_optimal_k,
    sort_spikes,
    evaluate_sorting,
    evaluate_os_per_cluster,
    run_sorting_pipeline,
    plot_sorting_summary,
    plot_k_search,
    # Sorting metrics
    isolation_distance,
    l_ratio,
    d_prime,
    d_prime_pairwise_matrix,
    peak_amplitude_snr,
    waveform_stability,
    amplitude_drift,
    fraction_missing,
    # Sorting plots
    plot_metric_bars,
    plot_d_prime_matrix,
    plot_waveform_stability,
    plot_amplitude_drift,
    plot_amplitude_histogram,
    # Zarr export/import
    to_zarr_flat,
    to_zarr_clustered,
    read_zarr_sorting,
)
from .tuning import (
    # utils
    steps2degree,
    # selectivity
    dosi_circular_normalised,
    circular_variance,
    gosi,
    gdsi,
    # tuning
    tuning_bandwidth,
    compute_f0_f1_f2,
    preferred_dori,
    get_os_metrics,
    OsMetricsResult,
    # fitting
    von_mises_fit,
    double_gaussian_fit,
    tuning_curve_interpolation,
    goodness_of_fit,
    # modulation
    modulation_ratio_per_orientation,
    cross_orientation_suppression,
    # temporal
    temporal_frequency_tuning,
    f1_phase,
    # population
    orientation_map_statistics,
    signal_correlations,
    noise_correlations,
    # statistics
    orientation_selectivity_significance,
    anova_across_orientations,
    bootstrap_ci,
    bootstrap_ci_strata,
    # tuning plots
    orientation_scatter_vm,
    plot_tuning_curve,
    plot_direction_polar,
    plot_f1f0_bars,
    plot_psth_by_orientation,
    plot_f1_phase,
    plot_population_orientation_histogram,
    plot_noise_correlation_matrix,
    plot_signal_correlation_matrix,
    plot_modulation_ratio,
    plot_temporal_frequency_tuning,
)
from .sta import (
    minimal_spike_train_analysis,
    calc_mfr_trial,
    plot_isi_histogram,
    plot_waveform_snippets,
    plot_spike_raster,
    # Spike train analyses
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
    # Spike train plots
    plot_autocorrelogram,
    plot_psth,
    plot_firing_rate_stability,
    plot_first_spike_latency,
    plot_trial_reliability_heatmap,
)
from ._utils import (
    circ_dist,
    circ_mean,
    guarded_divide,
    make_rng,
    wrap180,
    wrap360,
)

# Build __all__ from the names actually imported above.
# This avoids duplicating the list and stays in sync automatically.
import types as _types

__all__ = sorted(
    name
    for name, obj in vars().items()
    if not name.startswith("_")
    and not isinstance(obj, _types.ModuleType)
    and name not in ("__version__", "PackageNotFoundError")
) + ["__version__"]
