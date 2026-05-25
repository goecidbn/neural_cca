"""Orientation selectivity and tuning curve analysis."""

from .._utils import circ_dist, circ_mean, steps2degree, wrap180, wrap360
from .fitting import (
    double_gaussian_fit,
    goodness_of_fit,
    tuning_curve_interpolation,
    von_mises_fit,
)
from .modulation import (
    cross_orientation_suppression,
    modulation_ratio_per_orientation,
)
from .plotting import (
    orientation_scatter_vm,
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
)
from .population import (
    noise_correlations,
    orientation_map_statistics,
    signal_correlations,
)
from .selectivity import (
    circular_variance,
    dosi_circular_normalised,
    dsi_two_point,
    gdsi,
    gosi,
    osi_two_point,
)
from .statistics import (
    anova_across_orientations,
    bootstrap_ci,
    bootstrap_ci_strata,
    orientation_selectivity_significance,
)
from .temporal import (
    f1_phase,
    temporal_frequency_tuning,
)
from .tuning import (
    OsMetricsResult,
    compute_f0_f1_f2,
    get_os_metrics,
    preferred_dori,
    tuning_bandwidth,
)

__all__ = [
    # utils
    "steps2degree",
    "circ_dist",
    "circ_mean",
    "wrap180",
    "wrap360",
    # selectivity
    "dosi_circular_normalised",
    "circular_variance",
    "gosi",
    "gdsi",
    "osi_two_point",
    "dsi_two_point",
    # tuning
    "tuning_bandwidth",
    "compute_f0_f1_f2",
    "preferred_dori",
    "get_os_metrics",
    "OsMetricsResult",
    # fitting
    "von_mises_fit",
    "double_gaussian_fit",
    "tuning_curve_interpolation",
    "goodness_of_fit",
    # modulation
    "modulation_ratio_per_orientation",
    "cross_orientation_suppression",
    # temporal
    "temporal_frequency_tuning",
    "f1_phase",
    # population
    "orientation_map_statistics",
    "signal_correlations",
    "noise_correlations",
    # statistics
    "orientation_selectivity_significance",
    "anova_across_orientations",
    "bootstrap_ci",
    "bootstrap_ci_strata",
    # plotting
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
