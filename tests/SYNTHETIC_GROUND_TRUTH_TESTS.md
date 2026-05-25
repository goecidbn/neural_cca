# Synthetic Ground-Truth Tests — Plan & Initial Draft

This document is the working plan for the synthetic ground-truth test push
identified in the senior code review. It serves three purposes:

1. **Inventory** — every public metric in the package with its closed-form
   ground truth and the construction recipe needed to test it.
2. **Test draft** — copy-pasteable Python skeletons for each metric, ready to
   land in `tests/test_*.py` files.
3. **Tracking** — checkbox status per metric so the work can be split across
   PRs without losing context.

> **Why this exists**: the original example-based tests check that functions
> *return plausible numbers*; the regressions caught in the recent review
> (d-prime variance, OSI bootstrap, autocorrelogram zero-bin, LvR refractory
> suppression) would have been visible immediately under a closed-form regime.
> Each fix should ship with a closed-form regression test from this list.

---

## Coverage summary

| Subpackage             | Metrics | Closed-form covered today | After this push |
| ---------------------- | ------- | ------------------------- | --------------- |
| `sorting.metrics` | 12      | 1 (d_prime, just fixed)   | 12              |
| `sta.analysis`    | 16      | ~3 (partial)              | 16              |
| `tuning.selectivity` | 6 | ~2 (OSI, CirVar)          | 6               |
| `tuning.tuning` | 4     | ~2                        | 4               |
| `tuning.fitting` | 5    | ~2                        | 5               |
| `tuning.modulation` | 2 | 0                         | 2               |
| `tuning.temporal` | 2  | 0                         | 2               |
| `tuning.population` | 3 | ~1                       | 3               |
| `tuning.statistics` | 3 | ~1                       | 3               |
| **Total**              | **53**  | **~12 (23%)**             | **53 (100%)**   |

Numbers are approximate — the existing test files (`test_tuning.py`,
`test_tuning_extended.py`, …) contain a mix of analytic and behavioural
tests and the boundary is not always clean. The intent of this push is to
make every metric have **at least one** test where the expected value is a
closed-form expression rather than a "should be greater than X" assertion.

---

## Conftest fixtures (shared)

All tests share a small set of synthetic data generators. These belong in
`tests/conftest.py` so every test file can import them via fixtures.

```python
"""tests/conftest.py — shared synthetic data fixtures.

Conventions
-----------
- Every fixture takes a fixed seed (default 20260406) so tests are
  deterministic across runs and platforms.
- Stochastic fixtures use ``np.random.default_rng`` only.  No legacy
  ``np.random.seed`` calls anywhere.
- Closed-form parameters are exposed as fixture arguments so tests can
  pin the expected value alongside the construction.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Spike trains
# ---------------------------------------------------------------------------

@pytest.fixture
def regular_spike_train():
    """Periodic spikes at 100 Hz for 10 s.

    Closed-form metrics:
        - MFR        = 100.0 Hz
        - CV(ISI)    = 0
        - LV(ISI)    = 0
        - LvR(ISI)   = 0  (with default refractory_period)
        - Fano       = 0
    """
    rate = 100.0
    duration = 10.0
    spikes = np.arange(0.0, duration, 1.0 / rate, dtype=np.float64)
    return {"spikes": spikes, "rate": rate, "duration": duration}


@pytest.fixture
def poisson_spike_train():
    """Homogeneous Poisson process at 50 Hz for 60 s.

    Closed-form (asymptotic):
        - MFR  ≈ 50 Hz   (within ~0.5 Hz at this n)
        - CV   ≈ 1
        - Fano ≈ 1
    """
    rate = 50.0
    duration = 60.0
    rng = np.random.default_rng(20260406)
    n_expected = int(rate * duration)
    isis = rng.exponential(1.0 / rate, size=n_expected * 2)
    spikes = np.cumsum(isis)
    spikes = spikes[spikes < duration]
    return {"spikes": spikes, "rate": rate, "duration": duration}


@pytest.fixture
def gamma_spike_train():
    """Gamma-distributed ISIs with shape k=4, scale chosen for 50 Hz mean.

    Closed-form:
        - mean ISI = 1/50 s
        - CV       = 1/sqrt(k) = 0.5
    """
    rate = 50.0
    k = 4.0
    duration = 60.0
    scale = 1.0 / (rate * k)  # so that mean ISI = k * scale = 1/rate
    rng = np.random.default_rng(20260406)
    n_expected = int(rate * duration * 1.5)
    isis = rng.gamma(shape=k, scale=scale, size=n_expected)
    spikes = np.cumsum(isis)
    spikes = spikes[spikes < duration]
    return {"spikes": spikes, "rate": rate, "duration": duration, "k": k}


@pytest.fixture
def trial_spikes_regular():
    """Per-trial regular spikes.

    20 trials × 2.5 s, 5 spikes per trial in the post-stimulus window.
    Stimulus onset at 0.5 s.

    Closed-form:
        - calc_mfr_trial → 5 / 2.0 = 2.5 Hz for every trial
        - first_spike_latency mean = 0.05 s exactly
    """
    n_trials = 20
    trial_dur = 2.5
    stim_onset = 0.5
    n_per_trial = 5
    # Spikes at 0.55, 0.95, 1.35, 1.75, 2.15 in every trial
    rel = np.linspace(stim_onset + 0.05, stim_onset + 1.65, n_per_trial)
    spikes = np.tile(rel, n_trials)
    trials = np.repeat(np.arange(n_trials, dtype=np.int64), n_per_trial)
    return {
        "spikes": spikes, "trials": trials,
        "n_trials": n_trials, "trial_dur": trial_dur, "stim_onset": stim_onset,
        "expected_mfr": n_per_trial / (trial_dur - stim_onset),
        "expected_first_latency": 0.05,
    }


# ---------------------------------------------------------------------------
# Waveform clusters
# ---------------------------------------------------------------------------

@pytest.fixture
def two_unit_gaussians():
    """Two unit Gaussians in 8 dims, separated by 4 along axis 0.

    Closed-form:
        - d_prime           = 4.0 (per cluster)
        - isolation_distance: ~ chi2.ppf(0.5, df=8) translated by 4²
                              (test it by monotonicity, not exact)
    """
    rng = np.random.default_rng(20260406)
    n, dim = 5_000, 8
    delta = np.zeros(dim); delta[0] = 4.0
    X = np.vstack([
        rng.standard_normal((n, dim)),
        rng.standard_normal((n, dim)) + delta,
    ])
    labels = np.array([0] * n + [1] * n, dtype=np.int64)
    return {"X": X, "labels": labels, "expected_d_prime": 4.0}


@pytest.fixture
def synthetic_waveforms_with_noise():
    """Two waveform clusters: a known template + Gaussian noise.

    Closed-form:
        - est_snr ≈ (max(template) - min(template)) / (2 * noise_std)
        - peak_amplitude_snr ≈ max(|template|) / noise_std
    """
    rng = np.random.default_rng(20260406)
    n, snippet_len = 800, 38
    t = np.linspace(0, 1, snippet_len)
    template = -5.0 * np.exp(-((t - 0.3) ** 2) / 0.005)
    noise_std = 0.3
    waveforms = template + rng.normal(0, noise_std, (n, snippet_len))
    expected_est_snr = (template.max() - template.min()) / (2.0 * noise_std)
    expected_peak_snr = np.max(np.abs(template)) / noise_std
    return {
        "waveforms": waveforms, "template": template, "noise_std": noise_std,
        "expected_est_snr": expected_est_snr,
        "expected_peak_snr": expected_peak_snr,
    }


# ---------------------------------------------------------------------------
# Tuning curves
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_tuned_rates():
    """Mean firing rates from a Gaussian-tuned cell at 12 orientations.

    Closed-form:
        - Preferred orientation = 90.0 deg
        - tuning_bandwidth      = 20 * sqrt(2 ln 2) ≈ 23.548 deg (HWHH)
        - Rates at preferred = peak; at preferred ± 90 = baseline
    """
    pref = 90.0
    sigma = 20.0
    peak = 30.0
    base = 2.0
    angles = np.linspace(0.0, 360.0, 12, endpoint=False)
    diff = ((angles - pref + 90.0) % 180.0) - 90.0
    rates = base + (peak - base) * np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    return {
        "angles": angles, "rates": rates,
        "preferred": pref, "sigma": sigma,
        "peak": peak, "base": base,
        "expected_hwhh": sigma * np.sqrt(2.0 * np.log(2.0)),
    }


@pytest.fixture
def harmonic_psth():
    """PSTH with known F0/F1/F2 amplitudes.

    Closed-form (analytical signal at 100 Hz sampling, 2 s, f_stim = 2 Hz):
        - F0 (DC)           = 5.0
        - F1 amplitude      = 8.0
        - F2 amplitude      = 1.0
        - F1 phase at t=0   = 0.0
    """
    f_stim = 2.0
    bin_size = 0.01
    duration = 2.0
    fs = 1.0 / bin_size
    A0, A1, A2 = 5.0, 8.0, 1.0
    t = np.arange(0.0, duration, bin_size)
    psth = A0 + A1 * np.cos(2.0 * np.pi * f_stim * t) + A2 * np.cos(2.0 * np.pi * 2.0 * f_stim * t)
    return {
        "psth": psth, "fs": fs, "f_stim": f_stim,
        "expected_F0": A0, "expected_F1": A1, "expected_F2": A2,
    }
```

---

## 1 — `sorting.metrics` (12 metrics)

Already covered by closed-form regression: `d_prime`, `d_prime_pairwise_matrix`
(see `tests/test_sorting_metrics.py::TestDPrime`).

### 1.1 `est_snr` — closed form via template + Gaussian noise

```math
\text{SNR}_{\text{closed}} = \frac{\max(\text{template}) - \min(\text{template})}{2 \cdot \sigma_{\text{noise}}}
```

```python
def test_est_snr_closed_form(synthetic_waveforms_with_noise):
    from neural_cca.sorting.metrics import est_snr
    fx = synthetic_waveforms_with_noise
    snr = est_snr(fx["waveforms"])
    assert snr == pytest.approx(fx["expected_est_snr"], rel=0.05)
```

- [ ] Implement `test_est_snr_closed_form`
- [ ] Add edge case: zero noise → `inf` (already covered by current code, regression-pin it)

### 1.2 `calc_weighted_snr`

```python
def test_calc_weighted_snr_two_equal_clusters(synthetic_waveforms_with_noise):
    """Two equally-sized clusters of identical SNR → returns that SNR."""
    from neural_cca.sorting.metrics import calc_weighted_snr
    fx = synthetic_waveforms_with_noise
    n = len(fx["waveforms"]) // 2
    labels = np.zeros(2 * n, dtype=np.int64)
    labels[n:] = 1
    wv = np.vstack([fx["waveforms"][:n], fx["waveforms"][:n]])  # same data twice
    val = calc_weighted_snr(wv, labels)
    expected = fx["expected_est_snr"]
    assert val == pytest.approx(expected, rel=0.05)
```

- [ ] Implement `test_calc_weighted_snr_two_equal_clusters`

### 1.3 `peak_amplitude_snr`

```python
def test_peak_amplitude_snr_closed_form(synthetic_waveforms_with_noise):
    from neural_cca.sorting.metrics import peak_amplitude_snr
    fx = synthetic_waveforms_with_noise
    val = peak_amplitude_snr(fx["waveforms"])
    assert val == pytest.approx(fx["expected_peak_snr"], rel=0.10)
```

- [ ] Implement `test_peak_amplitude_snr_closed_form`

### 1.4 `rpvs`

```python
def test_rpvs_exact_count():
    """Inject k sub-refractory pairs into a regular spike train."""
    from neural_cca.sorting.metrics import rpvs
    base = np.arange(0.0, 10.0, 0.01, dtype=np.float64)  # 10 ms ISIs
    # Inject 5 RPVs (sub-1 ms ISIs) by adding extra spikes 0.5 ms after some
    extra = base[:5] + 0.0005
    spikes = np.sort(np.concatenate([base, extra]))
    n = rpvs(spikes, refractory_period=0.001, relative=False, all_cluster=True)
    assert n == 5
    rel = rpvs(spikes, refractory_period=0.001, relative=True, all_cluster=True)
    assert rel == pytest.approx(5 / len(spikes), abs=1e-12)
```

- [ ] Implement `test_rpvs_exact_count`
- [ ] Add: `relative` flag with `total == 0` → 0.0
- [ ] Add: `all_cluster=True` with `labels` provided → cross-cluster ISIs excluded

### 1.5 `spikes_before_stimulus`

```python
def test_spikes_before_stimulus_exact_count():
    from neural_cca.sorting.metrics import spikes_before_stimulus
    spikes = np.array([0.1, 0.2, 0.3, 0.6, 0.9])
    labels = np.zeros(5, dtype=np.int64)
    n = spikes_before_stimulus(spikes, labels, stimulus_onset=0.5,
                                relative=False, all_cluster=True)
    assert n == 3
    frac = spikes_before_stimulus(spikes, labels, stimulus_onset=0.5,
                                  relative=True, all_cluster=True)
    assert frac == pytest.approx(0.6, abs=1e-12)
```

- [ ] Implement `test_spikes_before_stimulus_exact_count`

### 1.6 `neg_silhouette_score`

```python
def test_neg_silhouette_score_well_separated_zero(two_unit_gaussians):
    from neural_cca.sorting.metrics import neg_silhouette_score
    fx = two_unit_gaussians
    frac = neg_silhouette_score(fx["X"], fx["labels"], relative=True)
    # Two clean unit Gaussians at separation 4 → essentially no negative-sil
    assert frac < 0.005

def test_neg_silhouette_score_overlapping_high():
    from neural_cca.sorting.metrics import neg_silhouette_score
    rng = np.random.default_rng(20260406)
    n = 500
    X = np.vstack([rng.standard_normal((n, 4)), rng.standard_normal((n, 4)) + 0.1])
    lab = np.array([0] * n + [1] * n)
    frac = neg_silhouette_score(X, lab, relative=True)
    # Nearly identical clusters → silhouette is negative for ~half
    assert 0.30 < frac < 0.70
```

- [ ] Implement both

### 1.7 `isolation_distance`

```python
def test_isolation_distance_monotonic_in_separation():
    """Isolation distance is monotone in cluster separation."""
    from neural_cca.sorting.metrics import isolation_distance
    rng = np.random.default_rng(20260406)
    n, dim = 500, 6
    vals = []
    for sep in [1.0, 2.0, 4.0, 8.0]:
        delta = np.zeros(dim); delta[0] = sep
        X = np.vstack([rng.standard_normal((n, dim)),
                       rng.standard_normal((n, dim)) + delta])
        lab = np.array([0] * n + [1] * n)
        vals.append(isolation_distance(X, lab)[0])
    assert all(b > a for a, b in zip(vals, vals[1:])), vals

def test_isolation_distance_too_few_neighbours_nan():
    """When n_out < n_c, isolation distance is undefined."""
    from neural_cca.sorting.metrics import isolation_distance
    rng = np.random.default_rng(20260406)
    X = np.vstack([rng.standard_normal((100, 4)),
                   rng.standard_normal((10, 4))])
    lab = np.array([0] * 100 + [1] * 10)
    result = isolation_distance(X, lab)
    assert np.isnan(result[0])  # cluster 0 has 100 members, only 10 outsiders
```

- [ ] Implement both

### 1.8 `l_ratio`

```python
def test_l_ratio_well_separated_low(two_unit_gaussians):
    from neural_cca.sorting.metrics import l_ratio
    fx = two_unit_gaussians
    result = l_ratio(fx["X"], fx["labels"])
    for v in result.values():
        assert v < 0.05  # well-separated → near-zero contamination

def test_l_ratio_overlapping_higher():
    from neural_cca.sorting.metrics import l_ratio
    rng = np.random.default_rng(20260406)
    n, dim = 1_000, 4
    X = np.vstack([rng.standard_normal((n, dim)),
                   rng.standard_normal((n, dim)) + 0.5])
    lab = np.array([0] * n + [1] * n)
    sep = l_ratio(*two_unit_gaussians_factory())
    overlap = l_ratio(X, lab)
    assert min(overlap.values()) > min(sep.values())
```

- [ ] Implement both (factory helper to be added)

### 1.9 `waveform_stability`

```python
def test_waveform_stability_constant_template_one():
    """Same waveform throughout the recording → r ≈ 1."""
    from neural_cca.sorting.metrics import waveform_stability
    rng = np.random.default_rng(20260406)
    n = 500
    template = rng.standard_normal(38)
    wv = template + rng.normal(0, 0.05, (n, 38))
    spike_times = np.arange(n, dtype=np.float64) * 0.001
    r = waveform_stability(spike_times, wv)
    assert r > 0.99

def test_waveform_stability_drifting_low():
    """Waveform shape that morphs over time → r << 1."""
    from neural_cca.sorting.metrics import waveform_stability
    rng = np.random.default_rng(20260406)
    n = 500
    t = np.linspace(0, 1, 38)
    wv = np.empty((n, 38))
    for i in range(n):
        shift = int(10 * i / n)  # peak migrates by up to 10 samples
        wv[i] = np.roll(-5.0 * np.exp(-((t - 0.3) ** 2) / 0.005), shift)
        wv[i] += rng.normal(0, 0.1, 38)
    spike_times = np.arange(n, dtype=np.float64) * 0.001
    r = waveform_stability(spike_times, wv)
    assert r < 0.9
```

- [ ] Implement both (already partially covered, tighten tolerances)

### 1.10 `amplitude_drift`

```python
def test_amplitude_drift_linear_trend_one():
    """Linearly growing amplitudes → Spearman r = 1.0 exactly."""
    from neural_cca.sorting.metrics import amplitude_drift
    n = 200
    t = np.linspace(0, 1, 38)
    template = -5.0 * np.exp(-((t - 0.3) ** 2) / 0.005)
    scale = np.linspace(1.0, 3.0, n)[:, None]
    wv = template * scale  # noiseless
    r = amplitude_drift(wv)
    assert r == pytest.approx(1.0, abs=1e-12)

def test_amplitude_drift_constant_zero():
    """Constant amplitudes → Spearman r ≈ 0 (uncorrelated noise)."""
    from neural_cca.sorting.metrics import amplitude_drift
    rng = np.random.default_rng(20260406)
    n = 200
    template = -5.0 * np.ones(38)
    wv = template + rng.normal(0, 0.1, (n, 38))
    r = amplitude_drift(wv)
    assert abs(r) < 0.15
```

- [ ] Implement both

### 1.11 `fraction_missing`

```python
def test_fraction_missing_known_threshold():
    """Sample from N(μ, σ); set threshold ⇒ closed-form Φ((T-μ)/σ)."""
    from scipy import stats as sp_stats
    from neural_cca.sorting.metrics import fraction_missing
    rng = np.random.default_rng(20260406)
    n = 5_000
    snippet = 38
    mu, sigma = 5.0, 1.0
    amps = rng.normal(mu, sigma, n)
    # Build dummy waveforms with these p2p amplitudes
    wv = np.zeros((n, snippet))
    wv[:, 10] = -amps / 2
    wv[:, 20] = amps / 2  # so p2p = amps[i]
    expected = float(sp_stats.norm.cdf(amps.min(), loc=mu, scale=sigma))
    val = fraction_missing(wv, normality_warn=False)
    assert val == pytest.approx(expected, abs=0.005)

def test_fraction_missing_warns_on_bimodal():
    """Bimodal amplitude distribution → KS test triggers warning."""
    from neural_cca.sorting.metrics import fraction_missing
    rng = np.random.default_rng(20260406)
    n = 1_000
    snippet = 38
    amps = np.concatenate([rng.normal(2, 0.2, n // 2),
                            rng.normal(8, 0.2, n // 2)])
    wv = np.zeros((n, snippet))
    wv[:, 10] = -amps / 2
    wv[:, 20] = amps / 2
    with pytest.warns(RuntimeWarning, match="not normal"):
        fraction_missing(wv, normality_warn=True)
```

- [ ] Implement both

### 1.12 `d_prime` & `d_prime_pairwise_matrix`

✅ **Already implemented** in `tests/test_sorting_metrics.py::TestDPrime`:
- `test_regression_analytic_two_unit_gaussians`
- `test_regression_per_feature_variance_not_global`
- `test_pairwise_matrix_helper_symmetry`
- `test_zero_pooled_std_returns_nan`
- `test_plotting_uses_same_helper`

---

## 2 — `sta.analysis` (16 metrics)

### 2.1 `minimal_spike_train_analysis`

```python
class TestMinimalSpikeTrainAnalysis:
    def test_regular_train(self, regular_spike_train):
        from neural_cca.sta.analysis import minimal_spike_train_analysis
        fx = regular_spike_train
        result = minimal_spike_train_analysis(
            fx["spikes"],
            n_trials=1, trial_length=fx["duration"], stim_onset=0.0,
        )
        assert result["mfr"] == pytest.approx(fx["rate"], rel=0.01)
        assert result["cv"] == pytest.approx(0.0, abs=1e-9)
        assert result["lvr"] == pytest.approx(0.0, abs=1e-9)

    def test_poisson_train(self, poisson_spike_train):
        from neural_cca.sta.analysis import minimal_spike_train_analysis
        fx = poisson_spike_train
        result = minimal_spike_train_analysis(
            fx["spikes"],
            n_trials=1, trial_length=fx["duration"], stim_onset=0.0,
        )
        assert result["mfr"] == pytest.approx(fx["rate"], rel=0.05)
        assert result["cv"] == pytest.approx(1.0, abs=0.10)

    def test_gamma_train_cv(self, gamma_spike_train):
        from neural_cca.sta.analysis import minimal_spike_train_analysis
        fx = gamma_spike_train
        result = minimal_spike_train_analysis(
            fx["spikes"],
            n_trials=1, trial_length=fx["duration"], stim_onset=0.0,
        )
        assert result["cv"] == pytest.approx(1.0 / np.sqrt(fx["k"]), abs=0.05)

    def test_lvr_refractory_forwarded(self, regular_spike_train):
        """Regression: refractory_period must be forwarded into _compute_lvr."""
        from neural_cca.sta.analysis import minimal_spike_train_analysis
        fx = regular_spike_train
        r1 = minimal_spike_train_analysis(
            fx["spikes"], n_trials=1, trial_length=fx["duration"],
            stim_onset=0.0, refractory_period=0.001,
        )
        r2 = minimal_spike_train_analysis(
            fx["spikes"], n_trials=1, trial_length=fx["duration"],
            stim_onset=0.0, refractory_period=0.005,
        )
        # LvR for a regular train should differ when refractory differs
        assert r1["lvr"] != r2["lvr"]
```

- [ ] Implement all four

### 2.2 `calc_mfr_trial`

```python
def test_calc_mfr_trial_exact(trial_spikes_regular):
    from neural_cca.sta.analysis import calc_mfr_trial
    fx = trial_spikes_regular
    result = calc_mfr_trial(
        fx["spikes"], fx["trials"],
        duration=fx["trial_dur"], stimulus_onset=fx["stim_onset"],
        n_trials=fx["n_trials"],
    )
    assert all(v == pytest.approx(fx["expected_mfr"], abs=1e-12)
               for v in result.values())
```

- [ ] Implement

### 2.3 `isi_violation_rate`

```python
def test_isi_violation_rate_known_count():
    from neural_cca.sta.analysis import isi_violation_rate
    base = np.arange(0.0, 1.0, 0.01)         # 100 spikes, 10 ms ISIs
    extra = base[:3] + 0.0005                # 3 sub-1ms violations
    spikes = np.sort(np.concatenate([base, extra]))
    rate = isi_violation_rate(spikes, refractory_period=0.001)
    assert rate == pytest.approx(3.0 / (spikes.max() - spikes.min()), abs=1e-6)
    pct = isi_violation_rate(spikes, refractory_period=0.001, return_percentage=True)
    n_isis = len(spikes) - 1
    assert pct == pytest.approx(3.0 / n_isis * 100.0, abs=1e-6)
```

- [ ] Implement

### 2.4 `firing_rate_stability`

```python
class TestFiringRateStability:
    def test_constant_rate_zero_cv(self, trial_spikes_regular):
        from neural_cca.sta.analysis import firing_rate_stability
        fx = trial_spikes_regular
        result = firing_rate_stability(
            fx["spikes"], fx["trials"],
            window_size=0.5, stat="mean",
            trial_length=fx["trial_dur"],
        )
        # All windows with the same rate → cv_of_stat ≈ 0
        assert result["cv_of_stat"] == pytest.approx(0.0, abs=0.02)

    def test_lvr_refractory_forwarded(self, poisson_spike_train):
        """Regression: refractory_period must be forwarded to _compute_lvr."""
        from neural_cca.sta.analysis import firing_rate_stability
        fx = poisson_spike_train
        trials = np.zeros(len(fx["spikes"]), dtype=np.int64)
        r1 = firing_rate_stability(
            fx["spikes"], trials, stat="lvr",
            trial_length=fx["duration"], refractory_period=0.001,
        )
        r2 = firing_rate_stability(
            fx["spikes"], trials, stat="lvr",
            trial_length=fx["duration"], refractory_period=0.005,
        )
        assert not np.allclose(r1["values"], r2["values"], equal_nan=True)
```

- [ ] Implement both

### 2.5 `autocorrelogram`

```python
class TestAutocorrelogram:
    def test_periodic_train_peak_at_period(self):
        """Periodic spikes at 100 Hz → ACG peak at 10 ms."""
        from neural_cca.sta.analysis import autocorrelogram
        rate = 100.0
        spikes = np.arange(0.0, 5.0, 1.0 / rate)  # period = 10 ms
        lags, counts = autocorrelogram(spikes, bin_size=0.001, max_lag=0.05)
        peak_lag = lags[np.argmax(counts)]
        assert peak_lag == pytest.approx(0.01, abs=0.0015)

    def test_zero_lag_excluded(self):
        """No spike pair can land at lag 0 (loop excludes self-pairs)."""
        from neural_cca.sta.analysis import autocorrelogram
        spikes = np.array([0.0, 0.005, 0.010, 0.015])  # 5 ms ISIs
        lags, counts = autocorrelogram(spikes, bin_size=0.001, max_lag=0.02)
        zero_bin_idx = int(np.argmin(np.abs(lags)))
        assert counts[zero_bin_idx] == 0

    def test_symmetric_about_zero(self):
        """ACG must be symmetric about lag 0 (every diff has its mirror)."""
        from neural_cca.sta.analysis import autocorrelogram
        rng = np.random.default_rng(20260406)
        spikes = np.sort(rng.uniform(0, 5.0, 200))
        lags, counts = autocorrelogram(spikes, bin_size=0.001, max_lag=0.05)
        # Compare bin at +k ms with bin at -k ms (excluding the central bin)
        n = len(counts)
        # Symmetric pairs
        for i in range(n // 2):
            j = n - 1 - i
            if i == j:
                continue
            assert counts[i] == counts[j], (i, j, counts[i], counts[j])
```

- [ ] Implement all three (the *symmetric* test is a regression for the
      zero-bin bug fixed in the review)

### 2.6 `fano_factor`

```python
class TestFanoFactor:
    def test_poisson_one(self, poisson_spike_train):
        from neural_cca.sta.analysis import fano_factor
        fx = poisson_spike_train
        ff = fano_factor(fx["spikes"], trial_length=fx["duration"], bin_size=0.1)
        assert ff == pytest.approx(1.0, abs=0.20)

    def test_regular_zero(self, regular_spike_train):
        from neural_cca.sta.analysis import fano_factor
        fx = regular_spike_train
        ff = fano_factor(fx["spikes"], trial_length=fx["duration"], bin_size=0.1)
        assert ff < 0.05
```

- [ ] Implement both

### 2.7 `local_variation`

```python
def test_local_variation_regular_zero(regular_spike_train):
    from neural_cca.sta.analysis import local_variation
    lv = local_variation(regular_spike_train["spikes"])
    assert lv == pytest.approx(0.0, abs=1e-9)

def test_local_variation_poisson_one(poisson_spike_train):
    from neural_cca.sta.analysis import local_variation
    lv = local_variation(poisson_spike_train["spikes"])
    assert lv == pytest.approx(1.0, abs=0.10)
```

- [ ] Implement both

### 2.8 `cv_log_isi`

```python
def test_cv_log_isi_lognormal_known_cv():
    """Lognormal ISIs with known σ_log → CV(log ISI) = σ_log / |μ_log|."""
    from neural_cca.sta.analysis import cv_log_isi
    rng = np.random.default_rng(20260406)
    mu_log, sigma_log = -3.0, 0.5  # in log seconds
    isis = np.exp(rng.normal(mu_log, sigma_log, 5_000))
    spikes = np.cumsum(isis)
    val = cv_log_isi(spikes)
    expected = sigma_log / abs(mu_log)
    assert val == pytest.approx(expected, abs=0.05)

def test_cv_log_isi_near_zero_mean_returns_nan():
    """Regression: ISIs clustered near 1 s give |mean log ISI| → 0."""
    from neural_cca.sta.analysis import cv_log_isi
    rng = np.random.default_rng(20260406)
    isis = np.exp(rng.normal(0.0, 1e-15, 100))  # mean log ISI essentially 0
    spikes = np.cumsum(isis)
    val = cv_log_isi(spikes)
    assert np.isnan(val)
```

- [ ] Implement both (the second is a regression for the exact-zero-check fix)

### 2.9 `psth`

```python
def test_psth_uniform_rate_constant_bins():
    """Inject N spikes per trial uniformly → constant per-bin firing rate."""
    from neural_cca.sta.analysis import psth
    n_trials = 50
    n_per_trial = 25
    trial_dur = 2.5
    bin_size = 0.05
    rng = np.random.default_rng(20260406)
    spikes = []
    trials = []
    for t in range(n_trials):
        spikes.append(rng.uniform(0.0, trial_dur, n_per_trial))
        trials.append(np.full(n_per_trial, t, dtype=np.int64))
    spikes = np.concatenate(spikes)
    trials = np.concatenate(trials)
    centres, rate = psth(spikes, trials, bin_size=bin_size, trial_length=trial_dur)
    expected_rate = n_per_trial / trial_dur  # uniform → all bins equal
    assert np.mean(rate) == pytest.approx(expected_rate, rel=0.10)

def test_psth_bin_widths_uniform():
    """Regression: bin edges must be exactly evenly spaced (linspace, not arange)."""
    from neural_cca.sta.analysis import psth
    spikes = np.array([0.5, 1.0, 1.5, 2.0])
    trials = np.zeros(4, dtype=np.int64)
    centres, rate = psth(spikes, trials, bin_size=0.01, trial_length=2.5)
    diffs = np.diff(centres)
    assert np.allclose(diffs, diffs[0], rtol=1e-12)
```

- [ ] Implement both (the second is a regression for the np.arange edges fix)

### 2.10 `trial_to_trial_reliability`

```python
class TestTrialToTrialReliability:
    def test_identical_trials_one(self):
        """Same PSTH on every trial → reliability = 1.0."""
        from neural_cca.sta.analysis import trial_to_trial_reliability
        n_trials = 30
        rel = np.array([0.5, 0.7, 0.9, 1.1, 1.3])  # spike pattern
        spikes = np.tile(rel, n_trials)
        trials = np.repeat(np.arange(n_trials, dtype=np.int64), len(rel))
        r = trial_to_trial_reliability(spikes, trials, stat="psth",
                                        bin_size=0.05, trial_length=2.5)
        assert r == pytest.approx(1.0, abs=0.05)

    def test_random_trials_low(self):
        """Independent random trials → reliability ~ 0."""
        from neural_cca.sta.analysis import trial_to_trial_reliability
        rng = np.random.default_rng(20260406)
        n_trials = 30
        spikes = []
        trials = []
        for t in range(n_trials):
            ts = rng.uniform(0, 2.5, 5)
            spikes.append(ts)
            trials.append(np.full(5, t, dtype=np.int64))
        r = trial_to_trial_reliability(np.concatenate(spikes),
                                       np.concatenate(trials),
                                       stat="psth", bin_size=0.05, trial_length=2.5)
        assert abs(r) < 0.30

    def test_f1_phase_consistent(self):
        """All trials with same F1 phase → reliability = 1.0."""
        from neural_cca.sta.analysis import trial_to_trial_reliability
        # Build PSTHs by hand: each trial is exactly cos(2πft) phase 0
        # ... use a synthetic harmonic_psth-style construction per trial
        pytest.skip("Construct synthetic phase-locked spike trains")
```

- [ ] Implement first two; design the third with `harmonic_psth` fixture

### 2.11 `trial_to_trial_correlation_matrix`

```python
def test_trial_to_trial_correlation_matrix_identical_trials():
    """Same trial repeated → matrix of all 1.0 (diagonal + off-diagonal)."""
    from neural_cca.sta.analysis import trial_to_trial_correlation_matrix
    n_trials = 10
    rel = np.array([0.5, 0.7, 0.9, 1.1, 1.3])
    spikes = np.tile(rel, n_trials)
    trials = np.repeat(np.arange(n_trials, dtype=np.int64), len(rel))
    _, corr = trial_to_trial_correlation_matrix(
        spikes, trials, bin_size=0.05, trial_length=2.5,
    )
    valid = corr[~np.isnan(corr)]
    assert np.allclose(valid, 1.0, atol=1e-9)
```

- [ ] Implement

### 2.12 `first_spike_latency`

```python
def test_first_spike_latency_known(trial_spikes_regular):
    from neural_cca.sta.analysis import first_spike_latency
    fx = trial_spikes_regular
    result = first_spike_latency(
        fx["spikes"], fx["trials"], stim_onset=fx["stim_onset"],
    )
    assert result["mean"] == pytest.approx(fx["expected_first_latency"], abs=1e-12)
    assert result["frac_responsive"] == 1.0
```

- [ ] Implement

---

## 3 — `tuning.selectivity` (6 metrics)

### 3.1 `dosi_circular_normalised` (OSI)

```python
class TestOSI:
    def test_dirac_one(self):
        """All firing at one orientation → OSI = 1."""
        from neural_cca.tuning.selectivity import dosi_circular_normalised
        angles = np.linspace(0, 360, 12, endpoint=False)
        rates = np.zeros(12); rates[3] = 10.0
        assert dosi_circular_normalised(rates, angles) == pytest.approx(1.0, abs=1e-12)

    def test_uniform_zero(self):
        """Uniform rates → OSI = 0."""
        from neural_cca.tuning.selectivity import dosi_circular_normalised
        angles = np.linspace(0, 360, 12, endpoint=False)
        rates = np.full(12, 5.0)
        assert dosi_circular_normalised(rates, angles) == pytest.approx(0.0, abs=1e-12)

    def test_orientation_invariance(self, gaussian_tuned_rates):
        """Rotating all angles by k° rotates pref ori by k° but leaves OSI."""
        from neural_cca.tuning.selectivity import dosi_circular_normalised
        fx = gaussian_tuned_rates
        osi1 = dosi_circular_normalised(fx["rates"], fx["angles"])
        osi2 = dosi_circular_normalised(fx["rates"], (fx["angles"] + 30) % 360)
        assert osi1 == pytest.approx(osi2, rel=1e-10)
```

- [ ] Implement all three

### 3.2 `dosi_circular_normalised` (DSI)

```python
def test_dsi_dirac_one():
    from neural_cca.tuning.selectivity import dosi_circular_normalised
    angles = np.linspace(0, 360, 12, endpoint=False)
    rates = np.zeros(12); rates[3] = 10.0
    val = dosi_circular_normalised(rates, angles, direction_selectivity=True)
    assert val == pytest.approx(1.0, abs=1e-12)
```

- [ ] Implement

### 3.3 `circular_variance`

```python
def test_circular_variance_complement_of_osi(gaussian_tuned_rates):
    from neural_cca.tuning.selectivity import (
        circular_variance, dosi_circular_normalised
    )
    fx = gaussian_tuned_rates
    cv = circular_variance(fx["rates"], fx["angles"])
    osi = dosi_circular_normalised(fx["rates"], fx["angles"])
    assert cv == pytest.approx(1.0 - osi, abs=1e-12)
```

- [ ] Implement (already partially covered, double-check it's exact)

### 3.4 `gosi`

```python
class TestGosi:
    def test_known_pref_orth_ratio(self):
        """R_pref=1, R_orth=0.3 → gOSI = 0.7/1.3 ≈ 0.538."""
        from neural_cca.tuning.selectivity import gosi
        angles = np.array([0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330.])
        rates = np.full(12, 0.3)
        rates[0] = 1.0          # preferred at 0°
        rates[3] = 0.3          # orthogonal at 90°
        rates[9] = 0.3          # orthogonal at 270°
        val = gosi(rates, angles)
        assert val == pytest.approx(0.7 / 1.3, abs=1e-12)

    def test_silent_neuron_returns_nan(self):
        """All-zero rates → undefined (NaN, not 0.0)."""
        from neural_cca.tuning.selectivity import gosi
        angles = np.linspace(0, 360, 12, endpoint=False)
        rates = np.zeros(12)
        assert np.isnan(gosi(rates, angles))
```

- [ ] Implement both

### 3.5 `gdsi`

```python
class TestGdsi:
    def test_known_pref_null_ratio(self):
        from neural_cca.tuning.selectivity import gdsi
        angles = np.linspace(0, 360, 12, endpoint=False)
        rates = np.full(12, 0.2)
        rates[0] = 1.0          # preferred at 0°
        rates[6] = 0.2          # null at 180°
        val = gdsi(rates, angles)
        assert val == pytest.approx(0.8 / 1.2, abs=1e-12)

    def test_silent_neuron_returns_nan(self):
        from neural_cca.tuning.selectivity import gdsi
        angles = np.linspace(0, 360, 12, endpoint=False)
        rates = np.zeros(12)
        assert np.isnan(gdsi(rates, angles))
```

- [ ] Implement both

### 3.6 `_rayleigh_test`

```python
def test_rayleigh_uniform_high_p():
    from neural_cca.tuning.selectivity import _rayleigh_test
    rng = np.random.default_rng(20260406)
    angles = rng.uniform(0, 2 * np.pi, 200)
    weights = np.ones(200)
    p = _rayleigh_test(angles, weights)
    assert p > 0.05  # cannot reject uniformity

def test_rayleigh_concentrated_low_p():
    from neural_cca.tuning.selectivity import _rayleigh_test
    rng = np.random.default_rng(20260406)
    angles = rng.normal(0, 0.1, 200)
    weights = np.ones(200)
    p = _rayleigh_test(angles, weights)
    assert p < 0.001
```

- [ ] Implement both

---

## 4 — `tuning.tuning` (4 metrics)

### 4.1 `tuning_bandwidth`

```python
def test_tuning_bandwidth_gaussian_hwhh(gaussian_tuned_rates):
    from neural_cca.tuning.tuning import tuning_bandwidth
    fx = gaussian_tuned_rates
    bw = tuning_bandwidth(fx["rates"], fx["angles"])
    assert bw == pytest.approx(fx["expected_hwhh"], rel=0.05)
```

- [ ] Implement

### 4.2 `compute_f0_f1_f2`

```python
def test_compute_f0_f1_f2_known_amplitudes(harmonic_psth):
    """Analytical signal A0 + A1*cos(2πft) + A2*cos(4πft) → exact F0/F1/F2."""
    from neural_cca.tuning.tuning import compute_f0_f1_f2
    fx = harmonic_psth
    F0, F1, F2 = compute_f0_f1_f2(fx["psth"], fx["fs"], fx["f_stim"])
    assert float(F0) == pytest.approx(fx["expected_F0"], rel=1e-3)
    assert float(F1) == pytest.approx(fx["expected_F1"], rel=1e-3)
    assert float(F2) == pytest.approx(fx["expected_F2"], rel=1e-3)


def test_compute_f0_f1_f2_2d_input_shape(harmonic_psth):
    """2-D PSTH input must return 1-D outputs (regression for test bug)."""
    from neural_cca.tuning.tuning import compute_f0_f1_f2
    fx = harmonic_psth
    psth_2d = np.tile(fx["psth"], (3, 1))
    F0, F1, F2 = compute_f0_f1_f2(psth_2d, fx["fs"], fx["f_stim"])
    assert F0.shape == (3,)
    assert F1.shape == (3,)
    assert F2.shape == (3,)
```

- [ ] Implement both (the second documents the shape contract)

### 4.3 `preferred_dori`

```python
def test_preferred_dori_dirac_recovers_angle():
    from neural_cca.tuning.tuning import preferred_dori
    angles = np.linspace(0, 360, 12, endpoint=False)
    for true_idx in range(12):
        rates = np.zeros(12); rates[true_idx] = 10.0
        # Doubled-angle convention: orientation maps 0..180 → 0..360 → /2
        # Direction maps 0..360 directly
        pref_dir = preferred_dori(rates, angles, direction_selectivity=True)
        assert pref_dir == pytest.approx(angles[true_idx], abs=1.0)
```

- [ ] Implement

### 4.4 `get_os_metrics`

```python
def test_get_os_metrics_tuned_neuron_recovers_preferred():
    """Build a synthetic tuned spike train; recover its preferred orientation."""
    from neural_cca.tuning.tuning import get_os_metrics
    # ... use the gaussian_tuned_rates fixture to build per-trial spike times
    # whose mean rate matches the rates vector, then run get_os_metrics
    pytest.skip("Wire up trial-level synthetic spike train builder")
```

- [ ] Implement (this is the integration test for the whole `get_os_metrics`
      pipeline; depends on a `gaussian_tuned_spikes` fixture that doesn't
      exist yet — see the `_TrialFilteredSpikes` refactor in the architecture
      section of the review)

---

## 5 — `tuning.fitting` (5 routines)

### 5.1 `goodness_of_fit`

```python
def test_goodness_of_fit_perfect_one():
    from neural_cca.tuning.fitting import goodness_of_fit
    obs = np.array([1.0, 2.0, 3.0])
    assert goodness_of_fit(obs, obs) == pytest.approx(1.0, abs=1e-12)

def test_goodness_of_fit_zero_variance_nan():
    from neural_cca.tuning.fitting import goodness_of_fit
    obs = np.array([5.0, 5.0, 5.0])
    pred = np.array([4.0, 5.0, 6.0])
    assert np.isnan(goodness_of_fit(obs, pred))
```

- [ ] Implement both

### 5.2 `von_mises_fit` (orientation mode)

```python
def test_von_mises_fit_orientation_recovers_known_params():
    """Sample noiseless orientation von Mises curve; recover its parameters."""
    from neural_cca.tuning.fitting import von_mises_fit
    R0_true, A_true, kappa_true, theta0_true_deg = 2.0, 8.0, 3.0, 45.0
    angles = np.linspace(0, 180, 18)
    theta = np.deg2rad(angles)
    rates = R0_true + A_true * np.exp(
        kappa_true * np.cos(2.0 * (theta - np.deg2rad(theta0_true_deg)))
    )
    result = von_mises_fit(rates, angles, tuning_type="orientation")
    assert result["tuning_type"] == "orientation"
    assert result["preferred_angle"] == pytest.approx(theta0_true_deg, abs=1.0)
    assert result["kappa"] == pytest.approx(kappa_true, rel=0.1)
    assert result["amplitude"] == pytest.approx(A_true, rel=0.1)
    assert result["baseline"] == pytest.approx(R0_true, abs=0.5)
    assert result["r_squared"] == pytest.approx(1.0, abs=1e-3)
```

- [ ] Implement

### 5.3 `von_mises_fit` (direction mode)

```python
def test_von_mises_fit_direction_recovers_two_bumps():
    """Direction-mode fit on a synthesised pref/null bump pair."""
    from neural_cca.tuning.fitting import von_mises_fit
    A_pref, A_null, kappa, theta0_deg, b = 8.0, 3.0, 2.0, 60.0, 1.0
    angles = np.linspace(0, 350, 36)
    theta = np.deg2rad(angles)
    theta0 = np.deg2rad(theta0_deg)
    rates = (
        A_pref * np.exp(kappa * np.cos(theta - theta0))
        + A_null * np.exp(kappa * np.cos(theta - (theta0 + np.pi)))
        + b
    )
    result = von_mises_fit(rates, angles, tuning_type="direction")
    assert result["tuning_type"] == "direction"
    assert result["preferred_angle"] == pytest.approx(theta0_deg, abs=2.0)
    assert result["amplitude_pref"] == pytest.approx(A_pref, rel=0.15)
    assert result["amplitude_null"] == pytest.approx(A_null, rel=0.15)
    assert result["r_squared"] > 0.99
```

- [ ] Implement

### 5.4 `double_gaussian_fit`

```python
def test_double_gaussian_fit_recovers_known_params():
    pytest.skip("Implement after double-Gaussian convention review")
```

- [ ] Implement (waits on the convention review)

### 5.5 `tuning_curve_interpolation`

```python
def test_tuning_curve_interpolation_finds_known_peak(gaussian_tuned_rates):
    from neural_cca.tuning.fitting import tuning_curve_interpolation
    fx = gaussian_tuned_rates
    pref = tuning_curve_interpolation(
        fx["rates"], fx["angles"], model="von_mises_orientation",
    )
    # Orientation model wraps to [0, 180); construction puts the peak at 90°
    assert pref % 180 == pytest.approx(fx["preferred"] % 180, abs=2.0)
```

- [ ] Implement

---

## 6 — `tuning.modulation` (2 metrics)

### 6.1 `modulation_ratio_per_orientation`

```python
def test_modulation_ratio_per_orientation_known_per_angle():
    """Inject a known DC + modulation per angle; verify F1/F0."""
    pytest.skip("Wire up per-angle PSTH builder")
```

- [ ] Implement (depends on per-angle PSTH builder fixture)

### 6.2 `cross_orientation_suppression`

```python
def test_cos_known_ratio():
    """R_pref=1, R_orth=0.3 → COS = 0.7."""
    from neural_cca.tuning.modulation import cross_orientation_suppression
    angles = np.linspace(0, 360, 12, endpoint=False)
    rates = np.full(12, 0.3)
    rates[0] = 1.0
    val = cross_orientation_suppression(rates, angles)
    assert val == pytest.approx(0.7, abs=1e-12)

def test_cos_silent_neuron_nan():
    """R_pref=0 → undefined (NaN, not 0.0)."""
    from neural_cca.tuning.modulation import cross_orientation_suppression
    angles = np.linspace(0, 360, 12, endpoint=False)
    rates = np.zeros(12)
    assert np.isnan(cross_orientation_suppression(rates, angles))
```

- [ ] Implement both

---

## 7 — `tuning.temporal` (2 metrics)

### 7.1 `f1_phase`

```python
def test_f1_phase_recovers_known_phase():
    from neural_cca.tuning.temporal import f1_phase
    f_stim = 2.0
    bin_size = 0.01
    duration = 4.0
    fs = 1.0 / bin_size
    t = np.arange(0.0, duration, bin_size)
    for true_phi in [-1.0, 0.0, 0.5, 1.5]:
        psth = 5.0 + 3.0 * np.cos(2 * np.pi * f_stim * t + true_phi)
        recovered = f1_phase(psth, fs, f_stim)
        # FFT phase convention: cos(2πft + φ) → -φ in this code's convention
        # Verify by computing the residual mod 2π
        diff = (recovered - (-true_phi) + np.pi) % (2 * np.pi) - np.pi
        assert abs(diff) < 0.05
```

- [ ] Implement

### 7.2 `temporal_frequency_tuning`

```python
def test_temporal_frequency_tuning_recovers_pref_tf():
    """Inject highest response at known TF; verify preferred_tf."""
    pytest.skip("Wire up per-TF spike train builder")
```

- [ ] Implement (depends on per-TF spike train builder)

---

## 8 — `tuning.population` (3 metrics)

### 8.1 `orientation_map_statistics`

```python
def test_orientation_map_uniform():
    from neural_cca.tuning.population import orientation_map_statistics
    rng = np.random.default_rng(20260406)
    oris = rng.uniform(0, 180, 200)
    result = orientation_map_statistics(oris)
    assert result["rayleigh_p"] > 0.05
    assert result["concentration"] < 0.2

def test_orientation_map_clustered():
    from neural_cca.tuning.population import orientation_map_statistics
    rng = np.random.default_rng(20260406)
    oris = (45.0 + rng.normal(0, 2.0, 200)) % 180
    result = orientation_map_statistics(oris)
    assert result["rayleigh_p"] < 1e-6
    assert result["mean_ori"] == pytest.approx(45.0, abs=1.0)
    assert result["concentration"] > 0.95
```

- [ ] Implement both

### 8.2 `signal_correlations`

```python
def test_signal_correlations_identical_one():
    from neural_cca.tuning.population import signal_correlations
    tc = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float64)
    corr = signal_correlations(tc)
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-12)

def test_signal_correlations_negated_minus_one():
    from neural_cca.tuning.population import signal_correlations
    tc = np.array([[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=np.float64)
    corr = signal_correlations(tc)
    assert corr[0, 1] == pytest.approx(-1.0, abs=1e-12)
```

- [ ] Implement both

### 8.3 `noise_correlations`

```python
def test_noise_correlations_independent_zero():
    from neural_cca.tuning.population import noise_correlations
    rng = np.random.default_rng(20260406)
    n_trials = 500
    angles = np.tile([0, 90], n_trials // 2)
    rates = rng.standard_normal((2, n_trials))
    corr = noise_correlations(rates, angles.astype(np.float64))
    assert abs(corr[0, 1]) < 0.1

def test_noise_correlations_identical_one():
    from neural_cca.tuning.population import noise_correlations
    rng = np.random.default_rng(20260406)
    n_trials = 200
    angles = np.tile([0, 90], n_trials // 2).astype(np.float64)
    base = rng.standard_normal((1, n_trials))
    rates = np.vstack([base, base])
    corr = noise_correlations(rates, angles)
    assert corr[0, 1] == pytest.approx(1.0, abs=1e-9)
```

- [ ] Implement both

---

## 9 — `tuning.statistics` (3 metrics)

### 9.1 `orientation_selectivity_significance`

```python
def test_orientation_significance_tuned_significant(gaussian_tuned_rates):
    from neural_cca.tuning.statistics import orientation_selectivity_significance
    fx = gaussian_tuned_rates
    result = orientation_selectivity_significance(
        fx["rates"], fx["angles"], n_permutations=500, rng_seed=20260406,
    )
    assert result["p_permutation"] < 0.05
    assert result["p_rayleigh"] < 0.05

def test_orientation_significance_uniform_not_significant():
    from neural_cca.tuning.statistics import orientation_selectivity_significance
    angles = np.linspace(0, 360, 12, endpoint=False)
    rates = np.full(12, 5.0)
    result = orientation_selectivity_significance(
        rates, angles, n_permutations=500, rng_seed=20260406,
    )
    assert result["p_permutation"] > 0.05
```

- [ ] Implement both

### 9.2 `anova_across_orientations`

```python
def test_anova_tuned_significant():
    pytest.skip("Wire up trial-level synthetic spike train builder")

def test_anova_uniform_not_significant():
    pytest.skip("Wire up trial-level synthetic spike train builder")
```

- [ ] Implement both (depends on per-trial fixture)

### 9.3 `bootstrap_ci`

```python
def test_bootstrap_ci_mean_known_width():
    """Bootstrap CI of mean(N(0,1)) with n=100 → width ≈ 2 * 1.96 * SE."""
    from neural_cca.tuning.statistics import bootstrap_ci
    rng = np.random.default_rng(20260406)
    data = rng.standard_normal(100)
    result = bootstrap_ci(data, np.mean, n_bootstrap=2000, rng_seed=20260406)
    se_expected = 1.0 / np.sqrt(100)
    width = result["ci_upper"] - result["ci_lower"]
    expected_width = 2 * 1.96 * se_expected
    assert width == pytest.approx(expected_width, rel=0.20)

def test_bootstrap_ci_handles_nan_iterations():
    """Bootstrap with stat_func that sometimes returns NaN must not produce NaN CI."""
    from neural_cca.tuning.statistics import bootstrap_ci
    rng = np.random.default_rng(20260406)
    data = rng.standard_normal(50)
    def stat(x):
        return float(np.mean(x)) if rng.random() > 0.1 else np.nan
    result = bootstrap_ci(data, stat, n_bootstrap=200, rng_seed=20260406)
    assert not np.isnan(result["ci_lower"])
    assert not np.isnan(result["ci_upper"])
```

- [ ] Implement both (the second is a regression for the NaN-handling fix)

### 9.4 `bootstrap_ci_strata`

`bootstrap_ci_strata` is the right tool when each trial firing rate is
paired with a stimulus angle and the statistic depends on that pairing
(OSI, DSI, gOSI, gDSI). It resamples *within* each angle stratum so the
joint distribution of `(rate, angle)` is preserved — plain `bootstrap_ci`
shuffles trials across angles and gives a meaningless null.

```python
def test_bootstrap_ci_strata_brackets_true_osi():
    """Stratified bootstrap CI of OSI brackets the true value
    for a tuned response, with width that shrinks roughly like
    1/sqrt(n_trials_per_angle)."""
    from neural_cca.tuning.statistics import bootstrap_ci_strata
    from neural_cca.tuning.selectivity import (
        dosi_circular_normalised,
    )
    rng = np.random.default_rng(20260406)
    angles_unique = np.linspace(0, 360, 12, endpoint=False)
    n_per = 30
    angles = np.repeat(angles_unique, n_per)
    true_rates = 2.0 + 18.0 * (
        np.cos(np.deg2rad(2 * (angles_unique - 90))) + 1.0
    ) / 2.0
    rates = np.repeat(true_rates, n_per) + rng.normal(0, 0.5, n_per * 12)
    result = bootstrap_ci_strata(
        rates, angles,
        lambda d, s: dosi_circular_normalised(d, s),
        n_bootstrap=400, rng_seed=20260406,
    )
    assert result["estimate"] > 0.1
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]
    assert (result["ci_upper"] - result["ci_lower"]) < 0.4

def test_bootstrap_ci_strata_keeps_pairing():
    """Resampled values must never cross stratum boundaries."""
    from neural_cca.tuning.statistics import bootstrap_ci_strata
    n_per = 5
    strata = np.repeat([10, 20, 30, 40], n_per)
    data = np.concatenate(
        [np.arange(s, s + n_per, dtype=float) for s in (10, 20, 30, 40)]
    )
    captured = {}
    def stat(d, s):
        captured["resampled"] = d.copy()
        return 0.0
    bootstrap_ci_strata(data, strata, stat, n_bootstrap=1, rng_seed=7)
    for s, lo in zip([10, 20, 30, 40], [10, 20, 30, 40]):
        mask = strata == s
        assert np.all((captured["resampled"][mask] >= lo)
                      & (captured["resampled"][mask] < lo + n_per))
```

- [ ] Implement both (regressions for the stratified-bootstrap fix; the
  first is the upgrade to the broken trial-shuffling bootstrap that
  used to live inside `get_os_metrics`).

---

## Migration plan

### Phase 1 — conftest + critical regressions (one PR)

- Create `tests/conftest.py` with all fixtures listed above.
- Land the **regression tests** for already-fixed bugs:
  - `test_lvr_refractory_forwarded` (LvR fix)
  - `test_cv_log_isi_near_zero_mean_returns_nan` (cv_log_isi fix)
  - `test_psth_bin_widths_uniform` (PSTH bin edges fix)
  - `test_autocorrelogram_symmetric_about_zero` (ACG zero-bin fix)
  - `test_gosi_silent_neuron_returns_nan`, `test_gdsi_silent_neuron_returns_nan`,
    `test_cos_silent_neuron_nan` (NaN-handling fix)
  - `test_bootstrap_ci_handles_nan_iterations` (bootstrap NaN fix)
  - `test_compute_f0_f1_f2_2d_input_shape` (the pre-existing test that was failing)

This PR is mechanical and pins every fix from the recent review.

### Phase 2 — `sorting.metrics` closed-form coverage (one PR)

Land sections 1.1–1.11 (1.12 already done). Each test pins a closed-form
expectation. ~12 new tests, all using the `synthetic_waveforms_with_noise` and
`two_unit_gaussians` fixtures.

### Phase 3 — `sta.analysis` closed-form coverage (one PR)

Land section 2 (~16 new tests). Adds the `regular_spike_train`,
`poisson_spike_train`, `gamma_spike_train`, `trial_spikes_regular` fixture
suite.

### Phase 4 — `tuning.*` closed-form coverage (one PR)

Land sections 3–9 (~25 new tests). Most are quick once the
`gaussian_tuned_rates` and `harmonic_psth` fixtures are in place. The handful
of tests that depend on a per-trial spike-train builder (4.4, 6.1, 7.2, 9.2)
should be left as `pytest.skip` and unblocked by the `_TrialFilteredSpikes`
refactor in the architecture roadmap.

### Phase 5 — `get_os_metrics` integration tests (separate PR)

Once `_TrialFilteredSpikes` lands (architecture roadmap item), wire up a
`gaussian_tuned_spikes` fixture that produces a trial-level spike train whose
mean firing rate per orientation matches `gaussian_tuned_rates`. Use it to
test:

- `get_os_metrics` recovers the preferred orientation
- `get_os_metrics(bootstrap_ci=True)` produces a CI that brackets the true OSI
- `get_os_metrics(compute_p_values=True)` reports `p_permutation < 0.05`
- Stratified bootstrap variance shrinks with `1/sqrt(n_trials)` (the cross-check
  for the OSI bootstrap fix)

---

## Tolerances — guidelines

| Test type                  | Tolerance                       | Rationale |
| -------------------------- | ------------------------------- | --------- |
| Closed-form, deterministic | `abs=1e-12` (or `rel=1e-10`)    | Pin exactly. Any drift is a real bug. |
| Closed-form, finite-sample | `rel=0.05`                      | Stochastic ground truth (Poisson, Gaussian sampling). |
| Curve-fit recovery         | `rel=0.10` for params, `abs=1.0` (deg) for angles | `curve_fit` has finite numerical precision |
| Asymptotic (Poisson CV→1)  | `abs=0.10` with n ≥ 1000        | Convergence rate ~1/sqrt(n) |
| Bootstrap CI width         | `rel=0.20`                      | Monte-Carlo variance |

If a test fails on a tolerance, **do not loosen the tolerance** without
checking whether the underlying answer is correct. Most "spurious" failures of
closed-form tests are real bugs that the previous tolerance was too loose to
catch.

---

## Status legend

- ✅ Implemented and passing
- 🟡 Stubbed in this document, not yet in `tests/`
- 🔵 Blocked on a fixture or refactor
- ❌ Implementation needed

Once a section moves from this document into `tests/`, replace its checkboxes
with status emoji and link to the actual file/line.
