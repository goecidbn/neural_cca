# Neural CCA — Common Neural CIDBN Analysis

<img src="./docs/_static/logo_neural_cca_wide.svg" alt="Logo" height="100" />


Minimal analysis package and repo for common analysis tasks on neural data from extracellular recordings.

## Subpackages

### `sorting` — Spike Sorting

KMeans-based spike clustering pipeline with automatic k-selection via silhouette analysis.

- **Sorting pipeline**: waveform loading, KM-based clustering, automatic cluster count selection, per-cluster orientation selectivity evaluation
- **Quality metrics**: silhouette score, SNR (per-cluster and weighted), refractory period violations (RPVs), isolation distance, L-ratio, d-prime, peak amplitude SNR, waveform stability, amplitude drift, fraction missing
- **Batch processing**: xarray/zarr-backed multi-recording pipelines with trial-angle mapping

### `sta` — Spike Train Analysis

Single-unit and multi-unit spike train statistics covering firing rate, variability, temporal structure, and trial-level reliability.

- **Firing rate**: mean firing rate (MFR), per-trial MFR, PSTH (peri-stimulus time histogram)
- **Variability**: coefficient of variation (CV), local variation (LvR), CV of log-ISI, Fano factor
- **Temporal structure**: autocorrelogram, ISI violation rate, first spike latency
- **Trial reliability**: trial-to-trial reliability (Pearson, Fano, F1 phase consistency), firing rate stability across recording segments

### `tuning` — Orientation & Direction Selectivity

Comprehensive orientation/direction tuning analysis for visual neuroscience experiments.

- **Selectivity indices**: OSI, DSI (circular vector-sum), gOSI, gDSI (ratio-based), circular variance — all with optional Rayleigh-test p-values
- **Curve fitting**: von Mises (with internal two-bump direction variant) and double Gaussian fits with R² goodness-of-fit and interpolated preferred orientation
- **Harmonic analysis**: F0/F1/F2 decomposition, F1 phase extraction, modulation ratio per orientation, simple/complex cell classification
- **Temporal frequency**: TF tuning curves with log-Gaussian fit, preferred TF, bandwidth
- **Cross-orientation**: suppression index (proxy from tuning curve)
- **Population**: orientation map statistics (Rayleigh test for uniformity), signal correlations (tuning curve similarity), noise correlations (trial-to-trial co-variability)
- **Statistical testing**: permutation + Rayleigh test for selectivity significance, one-way ANOVA across orientations, bootstrap confidence intervals
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
- [ ] potentially combine examples with the STA tools


## License

AGPL-3.0-only
