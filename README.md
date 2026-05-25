# Neural CCA — Common Neural CIDBN Analysis

<img src="./docs/_static/logo_neural_cca_wide.svg" alt="Logo" height="100" />


Minimal analysis package and repo for common analysis tasks on neural data from extracellular recordings.

## Scope

`neural_cca` operates on **already-detected, single-channel waveform
snippets** with paired spike times and trial / stimulus metadata.
It is intended for medium-scale recordings (a few channels, tens of
clusters, hundreds to low-thousands of trials) where the analysis
workflow is "load spikes, sort, characterise tuning". More powerful
internals — multi-channel template features, drift correction,
density-based clustering, spike-triggered receptive-field estimation —
are planned for future releases and will be additive (no API breakage
for the current single-channel pipeline). The scope statement here is
deliberately tool-agnostic: if your recording exceeds what
`neural_cca` handles today, any modern probe-scale sorter is
appropriate upstream of this package's tuning / quality / population
analyses.

## Subpackages

### `sorting` — Spike Sorting

KMeans-based spike clustering pipeline with automatic k-selection via silhouette analysis.

- **Sorting pipeline**: waveform loading, KM-based clustering, automatic cluster count selection, per-cluster orientation selectivity evaluation
- **Quality metrics**: silhouette score, SNR (per-cluster and weighted), refractory period violations (RPVs), isolation distance, L-ratio, d-prime, peak amplitude SNR, waveform stability, amplitude drift, fraction missing
- **Batch processing**: xarray/zarr-backed multi-recording pipelines with trial-angle mapping

### `spike_train` — Spike Train Statistics (previously `sta`)

Single-unit and multi-unit spike train statistics covering firing rate, variability, temporal structure, and trial-level reliability.

- **Firing rate**: mean firing rate (MFR), per-trial MFR, PSTH (peri-stimulus time histogram)
- **Variability**: coefficient of variation (CV), local variation (LvR), CV of log-ISI, Fano factor
- **Temporal structure**: autocorrelogram, ISI violation rate, first spike latency
- **Trial reliability**: trial-to-trial reliability (Pearson, Fano, F1 phase consistency), firing rate stability across recording segments

### `tuning` — Orientation & Direction Selectivity

Comprehensive orientation/direction tuning analysis for visual neuroscience experiments.

- **Selectivity indices**: vector-sum / "global" OSI, DSI, gOSI, gDSI (the modern Mazurek 2014 convention; gOSI ≡ 1 − circular variance), the explicit two-point Niell & Stryker (2008) ratios `osi_two_point` / `dsi_two_point`, and the circular variance (Ringach 2002).  Permutation-test significance via `orientation_selectivity_significance` (Phipson & Smyth 2010 corrected); rate-weighted Rayleigh available as a descriptive companion statistic.
- **Curve fitting**: von Mises (with internal two-bump direction variant) and double Gaussian fits with R² goodness-of-fit and interpolated preferred orientation
- **Harmonic analysis**: F0/F1/F2 decomposition, F1 phase extraction, modulation ratio per orientation, simple/complex cell classification
- **Temporal frequency**: TF tuning curves with log-Gaussian fit, preferred TF, bandwidth
- **Cross-orientation**: suppression index (proxy from tuning curve)
- **Population**: orientation map statistics (Rayleigh test for uniformity), signal correlations (tuning curve similarity), noise correlations (trial-to-trial co-variability)
- **Statistical testing**: permutation-test selectivity significance (Phipson & Smyth 2010 corrected, with a rate-weighted Rayleigh statistic returned alongside as a descriptive companion), one-way ANOVA across orientations, BCa-default bootstrap confidence intervals (`method="bca"` — Efron 1987, second-order accurate for boundary-bounded statistics like OSI/DSI)
- **Composite function**: `get_os_metrics()` 
— all-in-one metrics with optional fitting, p-values, gOSI/gDSI, and bootstrap CIs

### `synthetic` — Synthetic data generators

Single source of truth for the synthetic spike-train and waveform data used by the example notebooks and the test fixtures.

- **`poisson_train`**: bin-wise inhomogeneous Poisson with an absolute refractory period
- **`make_tuned_spikes`**: single Gaussian-tuned cluster — the helper backing `tests/conftest.py`
- **`make_two_unit_demo`**: the canonical two-unit demo (orientation-indifferent + orientation-selective cluster, Gaussian-template waveforms, ready-to-use `SortingData`) used at the top of every example notebook

## Installation

```bash
pip install neural-cca

# With batch processing support (xarray + zarr):
pip install neural-cca[batch]
```

## Quick Start

The example below loads synthetic spike data into the `SortingData` container, runs the full sorting pipeline (PCA → KMeans → automatic k-selection → per-cluster quality metrics and orientation selectivity), then prints a summary of cluster quality and OS metrics for each identified unit.

```python
from pprint import pprint
from neural_cca import load_from_arrays, run_sorting_pipeline, steps2degree

data = load_from_arrays(
    waveforms=waveforms,     # (n_spikes, snippet_length) float64
    spike_times=spike_times, # (n_spikes,) float64, trial-relative seconds
    trials=trials,           # (n_spikes,) int64
    angles=angles,           # (n_trials,) float64, degrees
)

result = run_sorting_pipeline(data)
pprint(result.quality)

for cl, m in result.os_metrics.items():
    print(f"Cluster {cl}: OSI={m['osi']:.3f}, Pref={m['preferred_orientation']:.1f}°")
```

See also the [example notebooks](./examples/) for a full walkthrough of all capabilities.

## Documentation

Full API reference at [GitHub Pages](https://goecidbn.github.io/neural_cca/).

## Roadmap

- [ ] Double Check Test Cases
- [ ] Add Export Options from CIDBN Zarr Reportings
- [ ] Add more realistic spike shapes for examples
- [ ] Add more example notebooks
  - [ ] add direction and "not only" orientation selective example
  - [ ] add inter trial variability
  - [ ] add example where spikes are not starting with stimulus onset
- [ ] potentially combine examples with the spike-train statistics module


## License

AGPL-3.0-only
