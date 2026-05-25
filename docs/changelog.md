# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-05-25

### Fixed

- `neural_cca/__init__.py`: `tomllib` is stdlib only on Python 3.11+;
  on 3.10 the pyproject-read fallback raised `ModuleNotFoundError` and
  silently reported `__version__ = "0.0.0"` for editable installs.
  Added a `tomli` backport import and a `tomli` runtime dependency
  guarded by `python_version < '3.11'`.
- `examples/example_sorting_pipeline.ipynb`: removed orphaned bernoulli-train
  code after `poisson_train`'s return — left over from a copy-paste, was a
  hard `SyntaxError` when the cell was parsed.
- `sta/analysis.py`: `trial_to_trial_reliability` docstring used `|R|`,
  which docutils interpreted as a substitution reference and Sphinx
  promoted to an error. Now `:math:`|R|``.
- `sorting/metrics.py`: `fraction_missing` docstring referenced a
  nonexistent `known_issues` page via `:doc:`. Inlined the alternative-
  estimator suggestion instead.
- `tests/test_tuning.py`: removed dead assignment to `angles` in
  `TestCalcMfrTrial::test_known_rate`.

### Removed

- `neural_cca/sorting/zarr_export.py`: 5-line backwards-compat shim that
  only re-exported `to_zarr_flat`, `to_zarr_clustered`,
  `read_zarr_sorting` from `io_util.py`. The filename suggested it
  owned the code; it didn't. No external importers found. Import the
  three functions from `neural_cca.sorting.io_util` (or from the
  top-level `neural_cca`) instead.
- `neural_cca/tuning/trial_rates.py`: empty deprecation stub. The
  `calc_mfr_trial` function moved to `neural_cca.sta.analysis` long
  ago; no external importers found.

### Documentation

- Removed the fictitious "sum-of-von-Mises" curve-fit claim from
  README and from the [0.1.0] changelog entry — no such function
  exists in `neural_cca.tuning.fitting`; the closest is the internal
  two-bump `_vm_direction_model` used inside `von_mises_fit`.
- `docs/conf.py`: silenced inherited `dict`-method autosummary stubs
  for the `OsMetricsResult` `TypedDict` subclass via
  `numpydoc_show_inherited_class_members = False` and
  `autodoc_default_options = {"inherited-members": False}`.
- `docs/api/sorting.rst`: split `SortingData` / `SortingResult` into a
  dedicated `containers` section and excluded them from `io_util`'s
  documented members to remove the duplicate-description warning.
- `sorting/io_util.py`: docstring previously pointed at the
  nonexistent `visioniceio.sorting_io.load_from_visioniceio`. Now
  points at `vision_ice_analysis.load_from_visioniceio` (the bridge).

### Internal

- `pyproject.toml` ruff: ignore `N803`/`N806` (scientific naming
  convention — `A`, `R0`, `X`, `Y`, …). Per-file ignores for
  `neural_cca/__init__.py` (re-export barrel), `examples/*.ipynb`
  (notebook reality), `tests/*.py` (long assertions,
  `pytest.importorskip` patterns).
- Auto-applied `ruff check --fix` (45 fixes, mostly import sort) and
  `ruff format` (30 files).
- Renamed bundled logos: `logo_mini_analysis_cidbn_wide.svg` →
  `logo_neural_cca_wide.svg`, `logo_mini_analysis.svg` →
  `logo_neural_cca.svg`; updated `docs/conf.py` reference.

## [0.1.0] - 2026-03-08

### Added

- Spike sorting pipeline (`run_sorting_pipeline`) with PCA + KMeans clustering
  and automatic k-selection via silhouette scores.
- Quality metrics: silhouette score, SNR, ISI violation rate, isolation distance,
  L-ratio, d-prime, peak amplitude SNR, waveform stability, amplitude drift,
  fraction missing.
- Orientation / direction selectivity: DOSI, gOSI, gDSI, circular variance,
  preferred orientation, tuning bandwidth.
- Tuning-curve fitting: von Mises (with internal two-bump direction
  variant) and double Gaussian, with interpolation and
  goodness-of-fit (R²).
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