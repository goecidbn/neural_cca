# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `neural_cca/synthetic.py`: new public module that owns all
  synthetic spike-train and waveform generation used by the four
  example notebooks and the test fixtures. Public helpers:
  `poisson_train` (binwise inhomogeneous Poisson with absolute
  refractory), `make_tuned_spikes` (single Gaussian-tuned cluster —
  the function `tests/conftest.py` used to inline), and
  `make_two_unit_demo` returning a `TwoUnitDemo` NamedTuple that
  bundles every array the canonical two-unit demo needs (spike
  times, trials, angles, waveforms, ground truth, and a loaded
  `SortingData`). All four example notebooks collapse from ~115
  lines of inline Poisson + Gaussian-template setup down to ~25
  lines of unpacking. `tests/conftest.py::make_tuned_spikes` now
  re-exports from this module so the test stream stays
  bit-identical (same `PCG64DXSM` + `SeedSequence(42)` construction).
- `docs/api/synthetic.rst`: dedicated API page for the new module.
- `sta/analysis.py`: `autocorrelogram` accepts `normalize="counts"`
  (default, unchanged) or `normalize="rate"` (divides by
  `n_spikes * bin_size`, returning Hz — matches the
  elephant / SpikeInterface convention). Empty spike trains return
  NaN-filled arrays under `rate` normalisation instead of dividing
  by zero.
- `sta/plotting.py`: `plot_autocorrelogram` forwards `normalize=`
  and labels the y-axis accordingly. Emits a `RuntimeWarning` when
  `refractory_period` is not a whole multiple of `bin_size` —
  in that regime the dashed refractory line lands inside a bar
  rather than on a bin edge and the plot can no longer be read as
  "everything left of the line is a violation".
- `tests/test_spike_train.py`: regression tests
  `test_rate_normalisation`, `test_invalid_normalize_raises`,
  `test_empty_spike_train_rate_is_nan`,
  `test_plot_warns_on_misaligned_refractory`,
  `test_plot_no_warning_when_aligned`.

### Changed

- `tests/test_tuning_statistics.py`: alphabetised the in-function
  imports in `TestEvaluateOsPerClusterRng::test_integer_seed_advances_across_clusters`
  to satisfy `ruff check` (I001).

### Fixed

- `sorting/sorting.py`: `run_sorting_pipeline` now forwards `rng` to
  `evaluate_os_per_cluster`, and `evaluate_os_per_cluster` materialises
  the RNG **once** via `make_rng` and shares it across clusters. Before,
  an integer seed produced identical bootstrap streams in every
  cluster (each `get_os_metrics` call rebuilt a fresh `Generator` from
  the same seed); the pipeline didn't pass `rng` down at all, so any
  downstream CI computed there was silently unseeded.
- `sorting/sorting.py`: `evaluate_sorting` accepts an optional
  `features` matrix. `run_sorting_pipeline` now preprocesses once and
  passes the resulting feature matrix to both clustering and quality
  evaluation, so `quality["silhouette_mean"]` agrees bit-for-bit with
  the value `k_search` recorded for the chosen `k`. Feature-space
  isolation metrics (`isolation_distance`, `l_ratio`, `d_prime`) move
  to that same space, matching the spike-sorting literature.
  Amplitude-based metrics (`snr_*`, `peak_amplitude_snr`,
  `waveform_stability`, `amplitude_drift`, `fraction_missing`) stay on
  raw waveforms — voltage amplitude is only meaningful there.
- `tuning/fitting.py`: `tuning_curve_interpolation` now samples the
  fitted model across one full period (180° for orientation, 360° for
  direction / double Gaussian) instead of `[angles.min(),
  angles.max()]`. The old behaviour returned the wrong preferred angle
  whenever the true peak fell across the wraparound (e.g. a direction
  cell preferring 350° on data sampled at `[0, 30, …, 330]`).
- `sta/analysis.py`: `autocorrelogram` vectorised the inner pair loop
  with `np.searchsorted` + two batched histogram calls. The previous
  implementation issued one `np.histogram` per pair (~O(n²) histogram
  calls). Pinned bin-for-bin equivalent to the naïve reference via
  `TestAutocorrelogram::test_vectorised_matches_naive`.
- `sorting/io_util.py`: zarr v3 renamed the per-array compressor
  argument from `compressor` to `compressors`. `_zarr_array` now
  translates explicit `compressor=` calls so users on zarr v3 stop
  crashing when supplying a codec.
- `tuning/selectivity.py`: `dosi_circular_normalised` now defaults
  `angles=None` (≡ `len(activities)`) and rejects an integer
  `angles` that doesn't match `len(activities)` with a clear
  `ValueError`. The previous magic default of `8` produced confusing
  shape errors for any other activity length.
- `tuning/_filter.py`: `_build_trial_filter` now validates that every
  trial ID lies in `[0, len(angles))`. The package-wide convention is
  that `angles[k]` is the stimulus angle of trial `k`, so
  out-of-range or negative trial IDs used to silently mis-map angles
  to rates.
- `sta/analysis.py`: `calc_mfr_trial` uses `max(trials) + 1` instead
  of `len(unique(trials))` when `n_trials is None`, so sparse trial
  IDs (e.g. `[0, 2, 5]`) no longer drop trials silently.

### Changed

- `sorting/sorting.py`: removed the unused `RngLike = "np.random..."`
  string from `__all__`. It was advertised as a type alias but was a
  bare runtime string — importing it gave a confusing literal back.
- `tuning/selectivity.py`, `tuning/modulation.py`: the orthogonal
  ±90° lookup in `gosi` and `cross_orientation_suppression` collapsed
  to a single `wrap180(pref + 90)` lookup. `+90` and `−90` fold to
  the same angle mod 180°, so the previous two-element average was
  algebraically identical to a single lookup.
- `sorting/metrics.py`: `peak_amplitude_snr`, `waveform_stability`,
  `amplitude_drift`, `fraction_missing` now use the shared
  `_validate_cluster_args` helper instead of an ad-hoc partial check.
  Passing `all_clusters=True` together with a `cluster_id` now
  raises (it was silently ignoring `cluster_id` before).
- `pyproject.toml`: ruff `target-version` lowered from `py312` to
  `py310` to match `requires-python`. The previous setting let ruff
  suggest syntax 3.10 users couldn't adopt.

### Added

- `sorting/__init__.py`, `neural_cca/__init__.py`,
  `docs/api/sorting.rst`: `batch_sort_experiment` is now re-exported
  from the package and documented under the sorting API.
- `tests/test_tuning_fitting.py`:
  `TestTuningCurveInterpolation::test_orientation_wraparound_near_180`
  and `test_direction_wraparound_near_360` pin the
  `tuning_curve_interpolation` wrap-around fix.
- `tests/test_spike_train.py`:
  `TestAutocorrelogram::test_vectorised_matches_naive` pins the
  vectorised `autocorrelogram` against the naïve reference.
- `tests/test_sorting_preprocess.py`:
  `TestPipelinePreprocess::test_pipeline_silhouette_matches_k_search`,
  `test_pipeline_forwards_rng_to_os_bootstrap`, and
  `test_feature_space_metrics_depend_on_preprocess` pin the
  pipeline-level silhouette / rng / feature-space invariants.
- `tests/test_tuning_statistics.py`:
  `TestEvaluateOsPerClusterRng`, `TestTrialIndexValidation`, and
  `TestDosiIntShorthand` pin the rng-share, trial-ID validation, and
  `dosi` int-shorthand contracts.

### Documentation

- `docs/index.rst`: the quick-start snippet used the wrong keyword
  names (`labels=`, `cluster=`, `result.labels`). Now uses
  `cluster_labels=`, `cluster_id=`, and `result.cluster_labels`,
  matching the real `get_os_metrics` signature.
- `sta/plotting.py`: replaced the misleading "avoid circular import"
  comment on the in-function `_per_trial_isis` import with the actual
  reason (underscore-private helper, deliberately not promoted to the
  module-top barrel).

### Internal

- `.github/workflows/tests.yml`: added Python 3.13 to the matrix
  (already advertised in `classifiers`).

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