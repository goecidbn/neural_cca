# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-08

### Added

- Spike sorting pipeline (`run_sorting_pipeline`) with PCA + KMeans clustering
  and automatic k-selection via silhouette scores.
- Quality metrics: silhouette score, SNR, ISI violation rate, isolation distance,
  L-ratio, d-prime, peak amplitude SNR, waveform stability, amplitude drift,
  fraction missing.
- Orientation / direction selectivity: DOSI, gOSI, gDSI, circular variance,
  preferred orientation, tuning bandwidth.
- Tuning-curve fitting: von Mises, double Gaussian, sum-of-von-Mises, with
  interpolation and goodness-of-fit (R²).
- F0/F1/F2 harmonic analysis, modulation ratios per orientation,
  cross-orientation suppression.
- Temporal-frequency tuning and F1-phase analysis.
- Population analysis: orientation map statistics, signal correlations,
  noise correlations.
- Statistical testing: permutation-based orientation-selectivity significance,
  ANOVA across orientations, bootstrap confidence intervals.
- Spike train statistics: MFR, CV, LvR, Fano factor, autocorrelogram, PSTH,
  trial-to-trial reliability, first-spike latency, ISI violation rate,
  firing-rate stability.
- Comprehensive plotting functions for all analyses.
- Zarr export/import with two layouts: flat `(n_spikes, ...)` and clustered
  `(n_clusters, max_spikes, ...)`.
- Composite `get_os_metrics` function that computes all orientation-selectivity
  metrics in one call.
- Example Jupyter notebook (`examples/example_sorting_pipeline.ipynb`)
  demonstrating the full analysis workflow.

### Known Issues

- **OSI bootstrap confidence intervals**: needs nested bootstrap per orientation.
- **Preferred orientation (population)** double check visualisation x-tick labels.
- **Von Mises fit** double check implementation.
- **Random number generators**: use use case instances and not global.
- **Label Assumptions**: Several functions assume trial IDs are contiguous 0..n-1 and iterate range(...) instead of actual trial IDs.