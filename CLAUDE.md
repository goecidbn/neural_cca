# CLAUDE.md — onboarding notes for Claude Code agents

This file orients a fresh agent on the `neural_cca` codebase.  It
covers the package layout, the few conventions a Python-fluent agent
will not infer from reading the code, and — most importantly — the
v0.2.0 behavioural changes that are *easy to miss* and would cause
silent breakage if assumed away.

## What the package is

`neural_cca` is a spike-sorting + spike-train statistics + tuning
analysis package for **already-detected, single-channel waveform
snippets** from extracellular recordings.  It's intended for
medium-scale recordings (a few channels, tens of clusters, hundreds
to low thousands of trials).  It is **not** a probe-scale sorter —
the pipeline is PCA → KMeans with silhouette-based *k*-selection,
and the README explicitly states that multi-channel templates,
drift correction, and STA-based receptive-field estimation are
planned for future releases.

The user maintains this package as their research workhorse.  Quality
metrics, OSI/DSI definitions, and the underlying conventions are
chosen to match what reviewers expect from V1 methods sections.

## Package layout

```
neural_cca/
├── __init__.py              # Top-level barrel re-exports
├── _utils.py                # circular stats, RNG factory, guarded_divide
├── synthetic.py             # poisson_train, make_two_unit_demo, ...
├── sorting/                 # PCA → KMeans pipeline, quality metrics, zarr I/O
├── tuning/                  # OSI/DSI/gOSI/gDSI, von Mises, F0/F1/F2, ...
├── spike_train/             # MFR, CV, LvR, ACG, PSTH, Fano  (was: sta)
└── sta/                     # Deprecation shim — re-exports spike_train
```

## Conventions a fresh agent must know

1. **Trial-relative seconds.** All spike-time arrays use *per-trial*
   timestamps (each trial starts at `t = 0`).  Pair every spike-time
   array with a `trials` array of trial indices.  Functions that
   compute ISIs route through `_per_trial_isis` so cross-trial
   spike pairs are **never** counted as ISIs.  Globally sorting
   trial-relative timestamps creates pseudo-ISIs that contaminate
   CV / LV / LvR / refractory rates — `np.diff(np.sort(spike_times))`
   on this data is a **bug**.

2. **NaN means "undefined", not "zero".**  Pearson `r` on a flat
   tuning curve, contamination rate on a 1-spike cluster, silhouette
   at `k = 1`, etc., all return `np.nan`.  Tests and callers
   distinguish "undefined" from "uncorrelated/no contamination".

3. **`PCG64DXSM` everywhere.**  RNGs are materialised via
   `_utils.make_rng()`.  This wraps `PCG64DXSM` (the DXSM output
   variant) to avoid the parallel-stream self-correlation in
   plain `PCG64` (numpy/numpy#16313).  Never construct
   `np.random.default_rng()` directly in this package.

4. **`scikit-learn` random_state coercion.**  sklearn needs a uint32
   `random_state` (`[0, 2**32)`), not a `Generator`.
   `sorting.sorting._as_seed()` always returns one: an `int` (incl. a
   ~128-bit master seed) is mixed down through `SeedSequence`
   (deterministic), and a `Generator` yields one drawn uint32
   (consumed-stream semantics).  **Never** hand a raw seed to sklearn —
   a 128-bit one raises `InvalidParameterError`.
   `tests/test_rng_policy.py` enforces this and bans `default_rng` /
   `RandomState` / legacy `np.random` / plain `PCG64` in package code.

5. **`stim_window = (onset, end)`.**  The trial spans
   `[0, stim_window[1]]`; `onset` separates spontaneous from
   stimulated activity.  `SortingData.__post_init__` raises if
   `end <= onset` — a typo here used to silently NaN out firing
   rates downstream.

## v0.1.3–v0.2.0 changes a fresh agent **will not infer** from skimming

These are the changes most likely to bite if assumed away.  The
`gosi`/`sta`/`bca` items below shipped in **v0.1.3**; the
batch-driver removal and the trial-aware / hardening items ship in
**v0.2.0**.  Read `docs/changelog.md` for the authoritative
per-version list.

### Semantic flip: `gosi` / `gdsi`

Before the [Unreleased] line, `gosi(responses, angles)` returned the
**two-point** Niell & Stryker (2008) ratio
`(R_pref - R_orth) / (R_pref + R_orth)`.

**Now `gosi` returns the vector-sum (1 − circular variance) form,**
the modern Mazurek 2014 convention.  The two-point ratio is now
`osi_two_point` / `dsi_two_point`.  This is a silent numeric
change for any caller of `gosi(...)`.  When debugging a number
mismatch against a v0.1.0–v0.1.2 reference, suspect this first.

### Subpackage rename: `sta` → `spike_train`

`neural_cca.sta` still works but emits a `DeprecationWarning` on
import — the actual code lives under `neural_cca.spike_train`.
The rename frees the `sta` name for the canonical *spike-triggered
average* (Schwartz 2006) meaning.  When writing new code, use
`neural_cca.spike_train`.  When fixing bugs in old code, the shim
is fine.

### Default change: `bootstrap_ci(method="bca")`

`bootstrap_ci()` and `bootstrap_ci_strata()` now default to
bias-corrected and accelerated (BCa, Efron 1987) confidence
intervals instead of the plain percentile method.  The returned
dict carries a `"method"` key indicating which CI was actually
used (BCa falls back to `"percentile"` on degenerate bootstrap
distributions).  Tests that pin `ci_lower` / `ci_upper` to
specific decimals will fail under the new default — use
`method="percentile"` explicitly if you need v0.1.0–v0.1.2 numbers.

### Default change: `tuning_bandwidth(method="von_mises")`

The default bandwidth estimator is now the circular von Mises HWHH
(Swindale 1998), which is unbiased near orientations of 0°/180°.
The legacy linear-Gaussian fit is available via
`method="gaussian"` but is documented as biased near wraparound.

### Behaviour: `orientation_selectivity_significance` now uses
Phipson & Smyth (2010) `+1` correction; the smallest reportable
permutation `p` is `1 / (n_permutations + 1)` instead of `0`.

### Behaviour: `temporal_frequency_tuning(TF=0, ...)` no longer
contaminates the log-Gaussian fit.  TF=0 ("blank") trials are
stripped before fitting and their mean response is reported
separately as `baseline_response`.

### Stance: rate-weighted Rayleigh on tuning curves is descriptive,
not calibrated.  The `p_value=True` paths on
`dosi_circular_normalised` / `circular_variance` / `gosi` / `gdsi`
/ `osi_two_point` / `dsi_two_point` still return a `p_value` key,
but the docstrings now point callers at
`orientation_selectivity_significance(...)` for the
V1-literature-standard **permutation** test.  `is_significant` in
that function is now decided by the permutation `p` alone (the
Rayleigh is still reported, but is descriptive).

### Removed: `batch_sort_experiment` (moved to the bridge)

`neural_cca.sorting.batch` is **gone**.  The directory/zarr-loading
batch driver imported `visioniceio` directly — a leaf→leaf dependency
the working-tree architecture forbids — so it now lives in the
`vision_ice_analysis` bridge.  `from neural_cca import
batch_sort_experiment` no longer resolves; loop `run_sorting_pipeline`
yourself for multi-electrode runs, or call the bridge.
`tests/test_architecture.py` enforces "no `visioniceio` import inside
`neural_cca`" so the coupling cannot silently return.

## Test & lint commands

```bash
python3 -m pytest tests/ -q                  # full suite (~3.5 min)
python3 -m pytest tests/test_sorting_*.py -q # subset
python3 -m ruff check neural_cca tests       # lint
python3 -m ruff format neural_cca tests      # format (do this before commits)
```

The pytest config silences matplotlib's pyparsing deprecations; any
other warning during tests is intentional and should be inspected.

## Commit / release process

- Version lives in `pyproject.toml` (`project.version`) and is
  read by `neural_cca/__init__.py` via the `tomllib` fallback for
  editable installs.  `docs/conf.py` reads it from `pyproject.toml`
  too, so a version bump propagates automatically.
- The changelog (`docs/changelog.md`) follows Keep a Changelog
  with `[Unreleased]` at the top.  Stamp it on release.
- Use `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
  in commits made on behalf of the user.

## Files an agent commonly needs to read in the first few minutes

- `README.md` — public-facing scope and quick-start
- `docs/changelog.md` `[Unreleased]` — current state delta
- `neural_cca/__init__.py` — public surface
- `neural_cca/sorting/metrics.py` — all the quality-metric definitions
- `neural_cca/tuning/selectivity.py` — gosi / dsi / two-point conventions
- `neural_cca/tuning/statistics.py` — permutation, BCa bootstrap
- `neural_cca/spike_train/analysis.py` — ISI / ACG / PSTH

The reviewer scientist will ask "is this how Mazurek does it?  Is
this how Hill 2011 does it?" — when in doubt, the docstrings cite
the canonical reference.  Match it.
