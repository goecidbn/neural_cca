# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `sorting/batch.py`: `batch_sort_experiment()` now treats
  `stim_window`, `tlabel2angle`, `n_angle_steps`, and
  `stim_frequency` as **explicit** parameters rather than carrying
  Natal-specific defaults. `stim_window=None` and the combination
  `tlabel2angle=None` *and* `n_angle_steps=None` now raise
  `ValueError` with a pointer to the LabView 30-degree convention.
  `stim_frequency=None` (the new default) disables F1 / F0
  computation in the per-cluster tuning block. The eight sibling
  functions that still carry the legacy `(0.5, 2.5)` /
  `stim_frequency=2.0` defaults gained a one-line
  `# NOTE: Natal-specific default; v0.2.0 will make this required.`
  comment so the broader sweep is grep-findable (see the v0.2.0
  Roadmap section below).
- `sorting/metrics.py`: `rpvs()` gained a `trials` keyword that
  mirrors the B3 trial-aware pattern already used in
  `contamination_rate_hill()`. With `trials=` supplied, refractory
  violations are counted **within each trial** so cross-trial
  trial-relative spikes never sort into spurious sub-millisecond
  pseudo-ISIs. Omitting `trials` still works but now emits a
  `RuntimeWarning` flagging the possible inflation.
  `evaluate_sorting()` passes `trials=data.trials` automatically.
- `tuning/selectivity.py`: `dosi_circular_normalised()` replaces the
  silent NumPy `divide by zero` on all-zero activities with an
  explicit `RuntimeWarning` ("All activities zero; circular DOSI
  undefined - returning NaN.") and returns `np.nan`. The
  `p_value=True` path returns `{"value": nan, "p_value": ...}` so
  downstream tables stay schema-stable.
- `tuning/statistics.py`: `bootstrap_ci_strata()` emits a
  `RuntimeWarning` when the smallest stratum has fewer than 3
  observations, citing Mazurek et al. 2014 (≥ 5 repeats per
  condition) and noting that the BCa acceleration term becomes
  unstable on very small strata.
- `tuning/statistics.py:_bca_ci`: tightened the degenerate-jackknife
  guard from `den == 0.0` to `abs(den) < 1e-300` so subnormal
  denominators no longer slip through and amplify the acceleration
  term `a` to extreme values.

### Documentation

- `tuning/modulation.py`: documented that
  `modulation_ratio_per_orientation()` reports
  **mean(F1) / mean(F0)**, per Skottun et al. 1991, *not*
  `mean(F1 / F0)`.
- `sorting/metrics.py`: added a `Notes` paragraph to
  `fraction_missing()` calling out the empirical-KDE estimator's
  structural floor of `0.5 / n` from the Gaussian kernel placed at
  `amp_min`, with the recommendation to prefer
  `method="gaussian"` or `"lognormal"` for small clusters
  (`n < 50`).
- `tuning/statistics.py`: clarified the Phipson & Smyth (2010) `+1`
  correction in `orientation_selectivity_significance()` — the `+1`
  in both numerator and denominator simulates *including the
  observed value in the null distribution*; the observed statistic
  is NOT inserted into `null_osis` itself.

### Public API

- `neural_cca/__init__.py`: re-exported five names that were
  already in `neural_cca.sorting.__all__` but missing from the
  top-level barrel: `neg_silhouette_score`, `spikes_before_stimulus`,
  `est_snr`, `calc_weighted_snr`, `rpvs`.

### Packaging

- `pyproject.toml`: added `readme = "README.md"`,
  `Development Status :: 3 - Alpha`, `Topic :: Scientific/Engineering`
  classifiers, and a `Changelog` URL in `[project.urls]`.

### CI

- All three workflows (`tests.yml`, `lint.yml`, `docs.yml`) gained
  a `concurrency` block (cancel-in-progress on the same ref) and
  `setup-python@v5` now uses `cache: 'pip'` with
  `cache-dependency-path: pyproject.toml`.
- `lint.yml` pins `ruff==0.15.14` to match the bridge.

### Roadmap (v0.2.0)

- Move directory→Dataset loading out of `sorting/batch.py` and into
  the bridge (`vision_ice_analysis`); `batch_sort_experiment` to
  accept only `xr.Dataset` or a zarr path.
- Sweep remaining Natal-specific defaults
  (`stim_window=(0.5, 2.5)`, `stim_frequency=2.0`) from 8 sibling
  files: `sorting/containers.py`, `tuning/temporal.py`,
  `tuning/tuning.py`, `spike_train/analysis.py` (2 sites),
  `tuning/_filter.py`, `tuning/statistics.py`, `tuning/modulation.py`,
  `sorting/io_util.py`.
- Add `CITATION.cff` and Zenodo–GitHub integration for software DOI.
- Author `paper.md` for JOSS submission.
- Add `Topic :: Scientific/Engineering :: Bio-Informatics` once
  spike-sorting/electrophysiology scope is firm.

## [0.1.3] - 2026-05-26

### Added

- `sorting/metrics.py`: `contamination_rate_hill()` — the Hill et al.
  (2011) closed-form contamination-fraction estimator
  (`C = 0.5 * (1 - sqrt(1 - 2*Nv*T / (N²*(t_r - t_c))))`).
  Calibrated to the false-positive contamination percentage that
  reviewers expect when methods sections quote "contamination < X %".
  Wired into `evaluate_sorting` so the pipeline now reports both the
  descriptive `rel_rpvs` (SpikeInterface-style violations per spike)
  and the Hill contamination fraction per cluster.  Registered in
  `QUALITY_METRIC_KINDS` so zarr export round-trips it.
  Reference: doi:10.1523/JNEUROSCI.0971-11.2011.
- `tuning/selectivity.py`: `osi_two_point()` / `dsi_two_point()` —
  explicit Niell & Stryker (2008) two-point ratios
  `(R_pref − R_orth) / (R_pref + R_orth)`.  The functions previously
  named `gosi` / `gdsi` implemented this metric; they have been
  reclaimed for the vector-sum / global form (see Changed).
  Reference: doi:10.1523/JNEUROSCI.0623-08.2008.
- `tuning/statistics.py`: BCa (bias-corrected and accelerated)
  bootstrap CI option on `bootstrap_ci()` and
  `bootstrap_ci_strata()`.  `method="bca"` is the new default
  because it is second-order accurate and transformation-respecting
  — the right choice for boundary-bounded statistics like OSI / DSI
  / gOSI / gDSI in `[0, 1]`.  Returns a `method` key indicating
  which CI was actually used (falls back to `"percentile"` for
  degenerate bootstrap distributions).  Reference: Efron (1987),
  doi:10.1080/01621459.1987.10478410.
- `spike_train/analysis.py`: `first_spike_latency_thresholded()` —
  response-detection latency that ignores trials with no
  above-baseline firing and caps the analysis window.  Reports
  baseline rate and detection threshold alongside the latency
  vector.  References: Reich et al. (2000), Smyth et al. (2003),
  Tovée et al. (1993).
- `sorting/metrics.py`: `fraction_missing(method=...)` accepts
  `"lognormal"` (fit on log-amplitudes — better-calibrated for
  the typical V1 lognormal amplitude distribution; Buzsáki &
  Mizuseki 2014) and `"empirical"` (non-parametric Gaussian-KDE
  tail).  The default remains `"gaussian"` (Hill 2011 / Allen
  Institute convention) for cross-paper comparability.
- `tuning/temporal.py`: `temporal_frequency_tuning()` now returns
  `baseline_response` (the mean amplitude on TF=0 blank trials)
  separately from the log-Gaussian fit.  The fit no longer
  extrapolates a 40-octave-below floor for TF=0 trials, which
  could blow up `bandwidth` / `preferred_tf` for cells with strong
  blank responses.  References: Foster et al. (1985), Hawken et
  al. (1996).
- `neural_cca/spike_train/` — new subpackage holding the
  spike-train statistics that lived under `neural_cca/sta/` until
  v0.2.0.  See Changed for the migration path.

### Changed

- **BREAKING (with shim): `neural_cca.sta` → `neural_cca.spike_train`.**
  The old import path remains as a deprecation shim that re-exports
  every public symbol and emits a `DeprecationWarning` on import.
  The rename frees the `sta` namespace for the canonical
  *spike-triggered average* (Schwartz et al. 2006) meaning of
  "STA" in computational neuroscience.  Migrate by replacing
  `neural_cca.sta` → `neural_cca.spike_train` in every import.
- **Behaviour change: `tuning.selectivity.gosi` / `gdsi`** now
  return the **vector-sum** (global) selectivity index — the
  modern V1-literature standard, equal to `1 - circular_variance`
  and computed via `dosi_circular_normalised`.  Previously they
  returned the two-point Niell & Stryker (2008) ratio, which is
  now exposed as `osi_two_point` / `dsi_two_point`.  The two
  numbers coincide only for cells whose tuning curve is a single
  cosine bump; in general they differ.  References: Mazurek, Kager
  & Van Hooser (2014) doi:10.3389/fncir.2014.00092, Ringach et
  al. (2002) doi:10.1523/JNEUROSCI.22-13-05639.2002.
- `tuning/tuning.py`: `tuning_bandwidth()` defaults to a circular
  von Mises HWHH (`method="von_mises"`), which is unbiased for
  preferred orientations near 0°/180°.  The previous linear-Gaussian
  path is available as `method="gaussian"` for reproducing legacy
  results but is documented as biased near wraparound.
  `get_os_metrics` consumes the new default automatically.
  Reference: Swindale (1998) doi:10.1007/s004220050411.
- `tuning/statistics.py`: `orientation_selectivity_significance()`
  applies the Phipson & Smyth (2010) `+1` correction to the
  permutation p-value (smallest value is now `1/(B+1)` instead of
  `0`).  `is_significant` now keys off the permutation test alone;
  `p_rayleigh` is still returned but is documented as a
  descriptive concentration statistic, *not* a calibrated tail
  probability for tuning significance (the rate-weighted Rayleigh
  is non-standard for tuning curves — use the permutation null).
  Reference: doi:10.2202/1544-6115.1585.
- `sorting/metrics.py`: `isolation_distance()` and `l_ratio()` gain
  a `mode="worst_pair"` option that reports the worst (nearest)
  per-other-cluster value rather than pooling all non-cluster
  spikes.  The modern best-practice convention from Sibille et
  al. (2024) and the Allen Institute `ecephys_spike_sorting`
  pipeline.  Default remains `mode="global"` for backwards
  compatibility.  Reference: doi:10.1523/ENEURO.0554-23.2024.
- `sorting/metrics.py`: `fraction_missing()` gains a `clamp_max`
  parameter (default `0.5`) applied **uniformly across all three
  methods** (`gaussian`, `lognormal`, `empirical`).  Previously
  only the empirical path was clamped, and the docstring incorrectly
  claimed the Gaussian / lognormal paths were also bounded.  Pass
  `clamp_max=None` to disable clamping for diagnostic /
  threshold-tuning workflows.  Values are unconditionally clamped
  from below at `0.0`.
- `sorting/batch.py`: `batch_sort_experiment()` accepts
  `**pipeline_kwargs` forwarded verbatim to
  `run_sorting_pipeline()` per electrode, so uncommon options
  (`min_silhouette`, `preprocess`, `pca_components`, …) can be
  set on the batch driver without inflating its signature.
- `neural_cca/__init__.py`: `make_tuned_spikes` is now re-exported
  at the package top level alongside `TwoUnitDemo`,
  `make_two_unit_demo`, and `poisson_train`, so the synthetic
  surface advertised in the README is reachable via the canonical
  import path.

### Fixed

- `sorting/metrics.py`: `contamination_rate_hill()` now accepts an
  optional `trials=` parameter; when given, refractory-violation
  counting iterates **within each trial** so globally-sorted
  trial-relative spike times can no longer create spurious
  cross-trial pseudo-ISIs that over-estimated contamination.
  `evaluate_sorting()` passes `trials=data.trials` automatically.
  Continuous-recording behaviour (`trials=None`) is unchanged.
  Aligns with the package-wide trial-aware ISI convention
  documented in `CLAUDE.md` §1 and used by `_per_trial_isis` in
  `spike_train/analysis.py`.
- `tuning/temporal.py`: `temporal_frequency_tuning().preferred_tf`
  now computes argmax over **positive TFs only**.  A blank trial
  (TF=0) is a spontaneous-activity sample, not a "0 Hz grating";
  the previous behaviour could report `preferred_tf=0` for
  spontaneously-active cells, advertising a stimulus that was
  never shown.  The blank-trial level is still surfaced separately
  as `baseline_response`.
- `tuning/statistics.py`: `_bca_ci()` returns `(NaN, NaN)` on every
  degenerate path so the outer `bootstrap_ci()` /
  `bootstrap_ci_strata()` correctly fall back to the percentile
  CI **and** relabel `"method": "percentile"`.  Previously the
  inner fallback called `_percentile_ci()` directly but left the
  outer `"method"` label as `"bca"`, so the user-visible provenance
  did not match the actual computation.

### Documented

- `tuning/temporal.py::f1_phase`: full phase-convention block
  (cosine convention, range `(-π, π]`, reference frame at trial
  onset, wraparound-safe averaging guidance) added to the
  docstring.  References: Skottun et al. (1991), Berens (2009).
- `tuning/population.py`: `signal_correlations` /
  `noise_correlations` / `orientation_map_statistics` gain
  bibliographic references (Cohen & Kohn 2011, Averbeck et al.
  2006, Bair et al. 2001, Mardia & Jupp 2000).
- `spike_train/analysis.py`: module-level docstring expanded with
  Shinomoto et al. (2009) for LvR and Holt et al. (1996) for
  CV/Fano; `fano_factor` gains Eden & Kramer (2010) and Nawrot et
  al. (2008).
- `_utils.py::make_rng`: cites O'Neill (2014) for PCG and the
  numpy/numpy#16313 thread that motivates the DXSM choice.
- `README.md`: scope statement clarifying the package is intended
  for already-detected, single-channel waveform snippets; more
  powerful internals (multi-channel templates, drift correction,
  STA-based receptive-field estimation) are planned for future
  releases and will be additive.
- `README.md`: install snippet switched from `pip install neural-cca`
  to `pip install git+https://github.com/goecidbn/neural_cca.git`
  (package is not yet on PyPI).  A new roadmap item tracks the
  PyPI publish; once that lands the snippet reverts to the bare
  `pip install neural-cca` form.
- `README.md`: API summary picks up `contamination_rate_hill`,
  `first_spike_latency_thresholded`, `fraction_missing(method=...)`,
  `isolation_distance` / `l_ratio` `mode="worst_pair"`, and the
  BCa default on `bootstrap_ci`.
- `tuning/statistics.py::_bca_ci`: one-line comment documenting
  the DiCiccio & Efron (1996, *Stat. Sci.* 11:189 eq. 2.13) sign
  convention for the acceleration term `diffs = jk_mean - jk_valid`
  — both signs give the same `a` because the numerator is
  sum-of-cubes and the denominator is sum-of-squares^(3/2), so
  future reviewers don't second-guess the sign against textbook
  write-ups that use the opposite convention.
- `sorting/sorting.py::run_sorting_pipeline`: the free-form
  "Single-cluster (``k = 1``) Mode" section header (which numpydoc
  flagged as Unknown) is now nested under a recognised `Notes`
  section.
- `CLAUDE.md` + `docs/api/sta.rst`: replaced premature "v0.2.0"
  language with the actual landing version (`v0.1.3`) and made
  the headline section title precise: *0.1.2 + [Unreleased]
  changes a fresh agent will not infer from skimming*.
- `docs/changelog.md`: the `[0.1.2]` entries that referenced
  `sta/analysis.py` / `sta/plotting.py` as file paths now point at
  `spike_train/...`, with a one-line note at the top of the section
  explaining the rename happened in `[Unreleased]`.

### Tests

- `tests/test_sorting_metrics.py::TestContaminationRateHill`:
  closed-form regression (zero violations, known `N_v`),
  cross-trial false-positive suppression (the new `trials=` path),
  per-cluster dict shape, and required-argument validation.
- `tests/test_sorting_metrics.py::TestFractionMissingMethods`:
  cover the `lognormal` and `empirical` paths, the new
  `clamp_max` parameter (including `None`-disables-upper-bound),
  and the input-validation guards.
- `tests/test_sorting_metrics.py::TestWorstPairMode`: pin the
  algorithmic invariants `iso_global ≤ iso_worst_pair` (mixing
  reaches the *n*-th distance earlier) and
  `lr_global ≥ lr_worst_pair` (sum ≥ max), plus
  worst_pair-identifies-overlapping-neighbour behaviour.
- `tests/test_spike_train.py::TestFirstSpikeLatencyThresholded`:
  recover a known 50 ms response latency from a synthetic burst,
  verify a silent baseline-only cell reports
  `frac_responsive < 0.15`, and exercise input validation.
- `tests/test_tuning_statistics.py::TestBootstrapCI`:
  `test_default_method_is_bca`,
  `test_bca_differs_from_percentile_on_skewed_data` (on a
  right-skewed exponential), and
  `test_bca_falls_back_to_percentile_on_degenerate` covering the
  `_bca_ci` NaN-sentinel relabelling fix above.

## [0.1.2] - 2026-05-25

> **Note**: file paths under ``sta/`` in the entries below were renamed
> to ``spike_train/`` in *[Unreleased]*; the historical paths have been
> updated to the current paths so links in this changelog stay valid.
> The deprecation shim at ``neural_cca/sta/`` re-exports the same
> public surface and emits a ``DeprecationWarning`` on import.

### Added

- `sorting/sorting.py`: `run_sorting_pipeline(n_clusters=1)` is now
  a first-class supported path for pre-isolated single-unit channels
  (Kilosort export, manual curation, "trust the channel" recordings).
  `evaluate_sorting` accepts k=1, fills `silhouette_mean`,
  `neg_silhouette_rel` and the per-cluster `isolation_distance` /
  `l_ratio` / `d_prime` entries with `np.nan`, and emits a single
  `RuntimeWarning` listing exactly which keys are NaN by
  construction. Amplitude, RPV, and OS metrics remain well-defined.
- `sorting/sorting.py`: `run_sorting_pipeline(... min_silhouette=...)`
  — soft auto-select fallback to k=1. When the best silhouette in
  `k_range` is below the threshold the pipeline declines to split
  the data rather than reporting arbitrary halves. `k_search` is
  still populated so the user can audit the search; the boolean
  `min_silhouette_triggered` flag is recorded in `metadata`.
- `neural_cca/synthetic.py`: new public module owning all synthetic
  spike-train and waveform generation used by the four example
  notebooks and the test fixtures. Public helpers: `poisson_train`
  (binwise inhomogeneous Poisson with absolute refractory),
  `make_tuned_spikes` (single Gaussian-tuned cluster — the function
  `tests/conftest.py` used to inline), and `make_two_unit_demo`
  returning a `TwoUnitDemo` NamedTuple that bundles every array the
  canonical two-unit demo needs (spike times, trials, angles,
  waveforms, ground truth, and a loaded `SortingData`). All four
  example notebooks collapse from ~115 lines of inline Poisson +
  Gaussian-template setup down to ~25 lines of unpacking.
  `tests/conftest.py::make_tuned_spikes` now re-exports from this
  module so the test stream stays bit-identical (same `PCG64DXSM` +
  `SeedSequence(42)` construction).
- `docs/api/synthetic.rst`: dedicated API page for the new module.
- `spike_train/analysis.py`: `autocorrelogram` accepts `normalize="counts"`
  (default, unchanged) or `normalize="rate"` (divides by
  `n_spikes * bin_size`, returning Hz — matches the
  elephant / SpikeInterface convention). Empty spike trains return
  NaN-filled arrays under `rate` normalisation instead of dividing
  by zero.
- `spike_train/plotting.py`: `plot_autocorrelogram` forwards `normalize=`
  and labels the y-axis accordingly. Emits a `RuntimeWarning` when
  `refractory_period` is not a whole multiple of `bin_size` — in
  that regime the dashed refractory line lands inside a bar rather
  than on a bin edge and the plot can no longer be read as
  "everything left of the line is a violation".
- `sorting/__init__.py`, `neural_cca/__init__.py`,
  `docs/api/sorting.rst`: `batch_sort_experiment` is now re-exported
  from the package and documented under the sorting API.
- `tests/test_sorting_preprocess.py`: new `TestSingleCluster` class
  covering `n_clusters=1` end-to-end (quality, OS metrics, zarr
  round-trip via both layouts), the `find_optimal_k` / pipeline
  `k_range` guard, and the `min_silhouette` fallback with both
  positive (triggers) and negative (doesn't trigger) cases. Also
  added `TestPipelinePreprocess::test_pipeline_silhouette_matches_k_search`,
  `test_pipeline_forwards_rng_to_os_bootstrap`, and
  `test_feature_space_metrics_depend_on_preprocess` to pin the
  pipeline-level silhouette / rng / feature-space invariants.
- `tests/test_spike_train.py`: regression tests
  `test_rate_normalisation`, `test_invalid_normalize_raises`,
  `test_empty_spike_train_rate_is_nan`,
  `test_plot_warns_on_misaligned_refractory`,
  `test_plot_no_warning_when_aligned`, and
  `TestAutocorrelogram::test_vectorised_matches_naive` (pins the
  vectorised ACG against the naïve reference).
- `tests/test_tuning_fitting.py`:
  `TestTuningCurveInterpolation::test_orientation_wraparound_near_180`
  and `test_direction_wraparound_near_360` pin the
  `tuning_curve_interpolation` wrap-around fix.
- `tests/test_tuning_statistics.py`: `TestEvaluateOsPerClusterRng`,
  `TestTrialIndexValidation`, and `TestDosiIntShorthand` pin the
  rng-share, trial-ID validation, and `dosi` int-shorthand
  contracts.
- `tests/test_synthetic.py`: new file giving the `neural_cca.synthetic`
  module its own coverage — `TestPoissonTrain` (refractory contract,
  zero/array rate profiles, seeded reproducibility), `TestMakeTunedSpikes`
  (default-seed stream, shapes/dtypes, peak-near-preferred), and
  `TestMakeTwoUnitDemo` (NamedTuple field set, array-shape coherence,
  merged-spike chronological ordering, ground-truth alignment, C1
  orientation-indifference CV < 0.20, C2 selectivity gOSI > 0.5,
  seed → identical arrays).  `TestBatchSortExperiment` covers the
  public re-export contract and the bogus-path
  `FileNotFoundError` (the function previously had zero direct tests).
- `tests/test_sorting_metrics.py`: `TestCalcWeightedSNR`
  (clean-multi-cluster numeric path, one-degenerate-cluster
  renormalisation warning, all-degenerate → NaN) and
  `TestRpvsValidation` (negative / zero `refractory_period` rejected)
  pin the new robustness contracts.
- `tests/test_sorting_preprocess.py::TestSortingDataValidation`:
  zero-duration / inverted / valid `stim_window` cases for the new
  construction-time check.
- `tests/test_tuning_population.py`:
  `TestSignalCorrelations::test_flat_tuning_returns_nan` and
  `TestNoiseCorrelations::test_zero_residual_neuron_returns_nan`
  pin the new "undefined-vs-zero" convention for correlation
  matrices.

### Changed

- `sorting/sorting.py`: `find_optimal_k` (and the pipeline-internal
  `_find_optimal_k_from_features`) now raise `ValueError` when any
  `k < 2` appears in `k_range`. Before, the call crashed deep
  inside sklearn's `silhouette_score`; the new message points at
  `run_sorting_pipeline(n_clusters=1)` and the `min_silhouette`
  fallback as the intended single-cluster paths.
- `sorting/sorting.py`: removed the unused `RngLike = "np.random..."`
  string from `__all__`. It was advertised as a type alias but was
  a bare runtime string — importing it gave a confusing literal
  back.
- `tuning/selectivity.py`, `tuning/modulation.py`: the orthogonal
  ±90° lookup in `gosi` and `cross_orientation_suppression`
  collapsed to a single `wrap180(pref + 90)` lookup. `+90` and `−90`
  fold to the same angle mod 180°, so the previous two-element
  average was algebraically identical to a single lookup.
- `sorting/metrics.py`: `peak_amplitude_snr`, `waveform_stability`,
  `amplitude_drift`, `fraction_missing` now use the shared
  `_validate_cluster_args` helper instead of an ad-hoc partial
  check. Passing `all_clusters=True` together with a `cluster_id`
  now raises (it was silently ignoring `cluster_id` before).
- `pyproject.toml`: ruff `target-version` lowered from `py312` to
  `py310` to match `requires-python`. The previous setting let ruff
  suggest syntax 3.10 users couldn't adopt.
- `tests/test_tuning_statistics.py`: alphabetised the in-function
  imports in
  `TestEvaluateOsPerClusterRng::test_integer_seed_advances_across_clusters`
  to satisfy `ruff check` (I001).
- `sorting/containers.py`: `SortingData.__post_init__` now rejects a
  `stim_window` with onset ≥ end at construction time.  A zero or
  negative stimulus duration used to silently divide by zero deep
  in `batch.py`'s per-trial firing-rate calculation, leaving the
  user with NaN rates and no breadcrumb pointing at the typo.
- `sorting/metrics.py`: `rpvs` now refuses non-positive
  `refractory_period`.  A negative value silently inverted the
  ``isi < refractory`` comparison and reported zero violations for
  every spike train — the new `ValueError` surfaces the input bug
  at the call site.
- `tuning/population.py`: `signal_correlations` and
  `noise_correlations` now return `np.nan` (not `0.0`) for the
  off-diagonal entries whenever either neuron has a zero-variance
  tuning curve or zero-variance residual.  Pearson's *r* is
  undefined in that case; the previous `0.0` conflated "undefined"
  with "uncorrelated" and disagreed with the package-wide
  undefined-vs-zero convention already used by `gosi`, `osi`, etc.

### Fixed

- `sorting/sorting.py`: `run_sorting_pipeline` now forwards `rng` to
  `evaluate_os_per_cluster`, and `evaluate_os_per_cluster`
  materialises the RNG **once** via `make_rng` and shares it across
  clusters. Before, an integer seed produced identical bootstrap
  streams in every cluster (each `get_os_metrics` call rebuilt a
  fresh `Generator` from the same seed); the pipeline didn't pass
  `rng` down at all, so any downstream CI computed there was
  silently unseeded.
- `sorting/sorting.py`: `evaluate_sorting` accepts an optional
  `features` matrix. `run_sorting_pipeline` now preprocesses once
  and passes the resulting feature matrix to both clustering and
  quality evaluation, so `quality["silhouette_mean"]` agrees
  bit-for-bit with the value `k_search` recorded for the chosen
  `k`. Feature-space isolation metrics (`isolation_distance`,
  `l_ratio`, `d_prime`) move to that same space, matching the
  spike-sorting literature. Amplitude-based metrics (`snr_*`,
  `peak_amplitude_snr`, `waveform_stability`, `amplitude_drift`,
  `fraction_missing`) stay on raw waveforms — voltage amplitude is
  only meaningful there.
- `tuning/fitting.py`: `tuning_curve_interpolation` now samples the
  fitted model across one full period (180° for orientation, 360°
  for direction / double Gaussian) instead of `[angles.min(),
  angles.max()]`. The old behaviour returned the wrong preferred
  angle whenever the true peak fell across the wraparound (e.g. a
  direction cell preferring 350° on data sampled at `[0, 30, …,
  330]`).
- `spike_train/analysis.py`: `autocorrelogram` vectorised the inner pair
  loop with `np.searchsorted` + two batched histogram calls. The
  previous implementation issued one `np.histogram` per pair
  (~O(n²) histogram calls). Pinned bin-for-bin equivalent to the
  naïve reference via
  `TestAutocorrelogram::test_vectorised_matches_naive`.
- `sorting/io_util.py`: zarr v3 renamed the per-array compressor
  argument from `compressor` to `compressors`. `_zarr_array` now
  translates explicit `compressor=` calls so users on zarr v3 stop
  crashing when supplying a codec.
- `tuning/selectivity.py`: `dosi_circular_normalised` now defaults
  `angles=None` (≡ `len(activities)`) and rejects an integer
  `angles` that doesn't match `len(activities)` with a clear
  `ValueError`. The previous magic default of `8` produced
  confusing shape errors for any other activity length.
- `tuning/_filter.py`: `_build_trial_filter` now validates that
  every trial ID lies in `[0, len(angles))`. The package-wide
  convention is that `angles[k]` is the stimulus angle of trial
  `k`, so out-of-range or negative trial IDs used to silently
  mis-map angles to rates.
- `spike_train/analysis.py`: `calc_mfr_trial` uses `max(trials) + 1`
  instead of `len(unique(trials))` when `n_trials is None`, so
  sparse trial IDs (e.g. `[0, 2, 5]`) no longer drop trials
  silently.
- `sorting/metrics.py`: `est_snr` and (transitively)
  `calc_weighted_snr` no longer return a bogus ~1e15-scale SNR
  when a cluster has identical waveforms.  `est_snr` compared
  `noise_std == 0` exactly; mean-subtraction on identical rows
  produces a residual std around 1e-15 that slipped past that
  guard, and the function returned an essentially-infinite SNR
  instead of `np.nan`.  Now the threshold is `1e-12 ⋅ max(sig_amp, 1)`
  (data-scale relative).  `calc_weighted_snr` then **excludes** any
  NaN-returning cluster from the weighted mean, renormalises the
  remaining weights, and emits a `RuntimeWarning` naming the
  culprit cluster(s).  Previously a single degenerate cluster
  silently poisoned the entire recording's `snr_weighted` with
  NaN; the new behaviour both surfaces the issue and recovers a
  partial answer from the well-behaved clusters.
- `sorting/metrics.py`: `d_prime_pairwise_matrix` replaces the
  exact `pooled_std == 0` guard with `pooled_std < 1e-12`.  Float
  rounding from `np.sqrt(tiny + tiny)` could otherwise slip
  through and produce a division-by-near-zero blow-up.
- `tests/test_sorting_preprocess.py::TestSingleCluster::test_k1_zarr_roundtrip`:
  the `np.isnan(silhouette_mean) or silhouette_mean is None`
  assertion failed every time the zarr-attrs path serialised the
  NaN as JSON `null`.  Python evaluates the `or` operands
  left-to-right, so `np.isnan(None)` was called first and raised
  `TypeError` before the short-circuit could help.  Reordered to
  `silhouette_mean is None or np.isnan(silhouette_mean)`; the test
  now passes against every numpy/zarr combination.
- `tests/test_sorting_preprocess.py::TestSingleCluster`: the k=1
  smoke tests previously only checked that result keys existed.
  They now also assert OSI/DSI live in `[0, 1]` (or are NaN),
  preferred orientation lives in `[0, 180°)` (or is NaN), and the
  zarr round-trip preserves `waveforms` / `spike_times` / `trials`
  / `angles` bit-for-bit against the original input.

### Documentation

- `docs/index.rst`: the quick-start snippet used the wrong keyword
  names (`labels=`, `cluster=`, `result.labels`). Now uses
  `cluster_labels=`, `cluster_id=`, and `result.cluster_labels`,
  matching the real `get_os_metrics` signature.
- `spike_train/plotting.py`: replaced the misleading "avoid circular import"
  comment on the in-function `_per_trial_isis` import with the actual
  reason (underscore-private helper, deliberately not promoted to the
  module-top barrel).
- `examples/example_sorting_pipeline.ipynb` and
  `examples/example_spike_sorting.ipynb`: new "Single-cluster mode
  (`n_clusters=1`)" subsection showing the pre-isolated single-unit
  channel path on the synthetic demo data — one warning-catching cell
  that splits the quality dict into numeric vs. NaN-by-construction
  entries and verifies the OS metrics still compute on the lone
  unit, plus a paired `min_silhouette=0.99` cell exercising the soft
  auto-fallback to k=1.
- `docs/api/sorting.rst`, `docs/api/sta.rst`, `docs/api/tuning.rst`,
  `docs/api/synthetic.rst`: each module page now opens with a
  narrative *Overview* + *Scientific scope* section before the
  auto-doc API reference, so a first-time reader gets the
  algorithmic story (clustering pipeline + quality-metric families
  for sorting; trial-aware ISI accounting and the metric catalogue
  for sta; selectivity / fitting / harmonic / population blocks for
  tuning; Poisson-with-refractory and the two-unit demo for
  synthetic) before being dropped into the function signatures.
- `docs/index.rst`: new "Institutions & Funding" block at the bottom
  of the landing page with CIDBN and Niedersächsisches Ministerium
  für Wissenschaft und Kultur (MWK) logos and links — mirrors the
  layout used by the sibling
  [`pynamicgain`](https://goecidbn.github.io/pynamicgain/) docs so
  CIDBN packages render their funding section consistently.
  `docs/_static/custom.css` carries the `.logo-grid` flexbox rules;
  `html_css_files = ["custom.css"]` registered in `conf.py`.
- `neural_cca/sorting/metrics.py`: expanded the
  `_ledoit_wolf_precision` docstring with the closed-form shrinkage
  formula, the small-:math:`n` / large-:math:`p` regime that makes
  it the right estimator for spike-sorting feature matrices, and
  primary references (Ledoit & Wolf 2004, Schmitzer-Torbert et al.
  2005, Hill et al. 2011).  Behaviour is unchanged.
- `neural_cca/synthetic.py`: rewrote the `make_tuned_spikes`
  docstring blurb that called it "test-friendly" without saying
  why.  It now states explicitly that the function does **not**
  enforce an absolute refractory (so seeded test counts stay
  stable even if `poisson_train`'s refractory logic evolves) and
  points readers at `make_two_unit_demo` for the refractory-
  respecting demo path.
- `neural_cca/sorting/metrics.py`: `est_snr`,
  `d_prime_pairwise_matrix`, and `rpvs` docstrings tightened so
  the Returns / Raises blocks match the recent epsilon and
  refractory-validation changes.  Previously each said "is zero"
  (`est_snr`, `d_prime_pairwise_matrix`) or "on invalid argument
  combinations" (`rpvs`) without surfacing the new data-scale
  thresholds or the explicit ``refractory_period > 0`` contract.
- `neural_cca/sorting/containers.py`: `SortingData.stim_window`
  attribute description now spells out the ``onset < end``
  constraint enforced in `__post_init__`.
- `neural_cca/tuning/population.py`: `noise_correlations` Returns
  block now documents the ``np.nan``-for-zero-residual behaviour
  (the implementation already did it; the docstring was silent).

### Internal

- `.github/workflows/tests.yml`: added Python 3.13 to the matrix
  (already advertised in `classifiers`).
- `pyproject.toml`: `[tool.pytest.ini_options]` gained a
  `filterwarnings` list silencing `DeprecationWarning` from
  `matplotlib` and `pyparsing` so the test output is no longer
  cluttered with ~12 `PyparsingDeprecationWarning`s the first
  time matplotlib's font-config parser fires.  These warnings
  come from external libraries and are not actionable from this
  package.

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
- `spike_train/analysis.py`: `trial_to_trial_reliability` docstring used `|R|`,
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