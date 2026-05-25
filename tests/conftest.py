"""Shared test fixtures for neural_cca.

Every fixture uses a *deterministic* RNG (seeded from a constant) so
tests are reproducible across machines and CI runs.  The seed constant
``SEED`` lives in one place; change it here to re-draw every synthetic
dataset at once.

Fixtures are grouped by the subpackage they support:

    Sorting    — waveform arrays, cluster labels, ``SortingData`` containers
    Spike-train — Poisson / regular / bursty / identical trial-relative spikes
    Tuning     — orientation-tuned firing rates and angles

Convention: fixtures that return structured data use a **dict** so
the test can access named fields (``fx["spike_times"]``) and the
fixture can carry analytic ground-truth values
(``fx["expected_d_prime"]``) alongside the data.  This keeps each
test's assertion on one line and makes the expected value traceable
to the fixture definition.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from numpy.random import PCG64DXSM, Generator, SeedSequence

from neural_cca.sorting.containers import SortingData, SortingResult

# ------------------------------------------------------------------ #
# Global seed                                                          #
# ------------------------------------------------------------------ #

SEED = 20260409


def _test_rng(seed: int = SEED) -> np.random.Generator:
    """Create a ``PCG64DXSM``-backed Generator with SeedSequence scrambling.

    Matches the production ``make_rng`` helper in ``_utils.py``.
    Using the same BitGenerator class in tests ensures that the
    random streams are identical for a given seed, so switching the
    production code to ``PCG64DXSM`` doesn't break any seeded test.
    """
    return Generator(PCG64DXSM(SeedSequence(seed)))


@pytest.fixture
def rng():
    """Fresh deterministic ``Generator`` for any test that needs randomness."""
    return _test_rng(SEED)


# ==================================================================
# SORTING FIXTURES
# ==================================================================


def make_two_clusters(
    n: int = 200,
    sep: float = 5.0,
    dim: int = 10,
    rng_seed: int = SEED,
) -> dict:
    """Two isotropic unit-variance Gaussian clusters.

    Closed-form d' = ``sep`` / 1.0 = ``sep`` when per-feature
    variance is 1.0 (standard normal).

    Returns dict with ``features``, ``labels``, ``separation``,
    ``n_per_cluster``, ``dim``.
    """
    rng = _test_rng(rng_seed)
    X0 = rng.standard_normal((n, dim))
    X1 = rng.standard_normal((n, dim)) + sep
    return {
        "features": np.vstack([X0, X1]),
        "labels": np.array([0] * n + [1] * n),
        "separation": sep,
        "n_per_cluster": n,
        "dim": dim,
    }


@pytest.fixture
def two_clusters():
    """Two well-separated Gaussian clusters (sep=5, dim=10, n=200 each)."""
    return make_two_clusters()


def make_overlapping_clusters(
    n: int = 200,
    dim: int = 10,
    rng_seed: int = SEED,
) -> dict:
    """Two nearly overlapping Gaussian clusters (separation 0.1)."""
    rng = _test_rng(rng_seed)
    X0 = rng.standard_normal((n, dim))
    X1 = rng.standard_normal((n, dim)) + 0.1
    return {
        "features": np.vstack([X0, X1]),
        "labels": np.array([0] * n + [1] * n),
        "separation": 0.1,
        "n_per_cluster": n,
        "dim": dim,
    }


@pytest.fixture
def overlapping_clusters():
    """Two nearly overlapping Gaussian clusters."""
    return make_overlapping_clusters()


def make_waveforms(
    n: int = 300,
    snippet_len: int = 38,
    noise_std: float = 0.5,
    rng_seed: int = SEED,
) -> dict:
    """Two-cluster waveforms: negative-going + positive-going sinusoidal
    templates plus Gaussian noise.

    Returns dict with ``waveforms``, ``labels``, ``template0``,
    ``template1``, ``noise_std``, ``n_per_cluster``,
    ``expected_snr0``, ``expected_snr1``.
    """
    rng = _test_rng(rng_seed)
    t = np.linspace(0, 1, snippet_len)
    t0 = -5.0 * np.sin(np.pi * t)
    t1 = 3.0 * np.sin(np.pi * t)
    wv0 = t0 + rng.standard_normal((n, snippet_len)) * noise_std
    wv1 = t1 + rng.standard_normal((n, snippet_len)) * noise_std
    return {
        "waveforms": np.vstack([wv0, wv1]).astype(np.float64),
        "labels": np.array([0] * n + [1] * n, dtype=np.int64),
        "template0": t0,
        "template1": t1,
        "noise_std": noise_std,
        "n_per_cluster": n,
        # Closed-form est_snr: (peak-to-peak of template) / (2 * noise_std)
        "expected_snr0": float((t0.max() - t0.min()) / (2 * noise_std)),
        "expected_snr1": float((t1.max() - t1.min()) / (2 * noise_std)),
    }


@pytest.fixture
def synthetic_waveforms():
    """Two-cluster sinusoidal waveforms with known SNR."""
    return make_waveforms()


def make_two_cluster_waveforms_only(
    n_per: int = 80,
    snippet_len: int = 32,
    noise_std: float = 0.4,
    rng_seed: int = SEED,
) -> np.ndarray:
    """Raw waveform array only (no labels / templates).

    Used by preprocessing / pipeline tests that only need the array.
    """
    rng = _test_rng(rng_seed)
    t = np.linspace(0, 1, snippet_len)
    ta = -5.0 * np.sin(np.pi * t)
    tb = +3.0 * np.sin(np.pi * t)
    wv_a = ta + rng.standard_normal((n_per, snippet_len)) * noise_std
    wv_b = tb + rng.standard_normal((n_per, snippet_len)) * noise_std
    return np.vstack([wv_a, wv_b]).astype(np.float64)


def make_sorting_data(
    n_per: int = 80,
    snippet_len: int = 32,
    rng_seed: int = SEED,
) -> SortingData:
    """Full ``SortingData`` container with two-cluster waveforms."""
    wv = make_two_cluster_waveforms_only(
        n_per=n_per,
        snippet_len=snippet_len,
        rng_seed=rng_seed,
    )
    n = wv.shape[0]
    rng = _test_rng(rng_seed)
    spike_times = rng.uniform(0.5, 2.5, n)
    trials = rng.integers(0, 12, n).astype(np.int64)
    angles = np.linspace(0, 330, 12)
    return SortingData(
        waveforms=wv,
        spike_times=spike_times,
        trials=trials,
        angles=angles,
        waveform_fs=32_000.0,
        n_trials=12,
        stim_window=(0.5, 2.5),
    )


@pytest.fixture
def sorting_data():
    """``SortingData`` container with two well-separated clusters."""
    return make_sorting_data()


@pytest.fixture()
def sample_zarr_data():
    """Minimal but realistic ``(SortingData, SortingResult)`` pair for
    zarr round-trip tests."""
    rng = _test_rng(42)
    n_spikes = 200
    snippet_length = 30
    n_trials = 24
    n_clusters = 3

    waveforms = rng.standard_normal((n_spikes, snippet_length))
    spike_times = rng.uniform(0.0, 2.5, size=n_spikes)
    trials = rng.integers(0, n_trials, size=n_spikes).astype(np.int64)
    angles = np.tile(np.linspace(0, 330, 12), 2)

    data = SortingData(
        waveforms=waveforms,
        spike_times=spike_times,
        trials=trials,
        angles=angles,
        waveform_fs=32_000.0,
        n_trials=n_trials,
        stim_window=(0.5, 2.5),
        stim_frequency=2.0,
        metadata={"electrode": 7, "animal": "mouse_01"},
    )

    cluster_labels = rng.integers(0, n_clusters, size=n_spikes).astype(np.int64)

    quality = {
        "neg_silhouette_rel": 0.05,
        "silhouette_mean": 0.42,
        "abs_rpvs": 3,
        "rel_rpvs": 0.015,
        "snr_weighted": 4.2,
        "snr_per_cluster": {0: 3.1, 1: 5.0, 2: 4.5},
        "isolation_distance": {0: 12.3, 1: 25.1, 2: 18.7},
        "l_ratio": {0: 0.08, 1: 0.02, 2: 0.05},
    }

    result = SortingResult(
        cluster_labels=cluster_labels,
        n_clusters=n_clusters,
        quality=quality,
        os_metrics={
            0: {"osi": 0.35, "preferred": 90.0},
            1: {"osi": 0.72, "preferred": 45.0},
            2: {"osi": 0.15, "preferred": 0.0},
        },
        k_search={2: 0.35, 3: 0.42, 4: 0.38, 5: 0.31},
        metadata={"rng": 0, "n_init": "auto"},
    )
    return data, result


# ==================================================================
# SPIKE-TRAIN FIXTURES
# ==================================================================


def make_regular_spikes(
    rate: float = 50.0,
    duration: float = 2.5,
    n_trials: int = 20,
) -> dict:
    """Clock-like spike train with constant ISI.

    Deterministic (no RNG).  CV = 0, LV = 0.
    """
    isi = 1.0 / rate
    times_per_trial = np.arange(0, duration, isi)
    return {
        "spike_times": np.tile(times_per_trial, n_trials),
        "trials": np.repeat(np.arange(n_trials), len(times_per_trial)),
        "rate": rate,
        "n_trials": n_trials,
        "trial_duration": duration,
        "expected_cv": 0.0,
        "expected_lv": 0.0,
    }


@pytest.fixture
def regular_spikes():
    """Clock-like 50 Hz spike train across 20 trials."""
    return make_regular_spikes()


def make_poisson_spikes(
    rate: float = 50.0,
    duration: float = 2.5,
    n_trials: int = 20,
    rng_seed: int = SEED,
) -> dict:
    """Poisson spike train.  CV ≈ 1, LV ≈ 1, Fano ≈ 1."""
    rng = _test_rng(rng_seed)
    all_times: list[npt.NDArray] = []
    all_trials: list[npt.NDArray] = []
    for t in range(n_trials):
        n = rng.poisson(rate * duration)
        times = np.sort(rng.uniform(0, duration, n))
        all_times.append(times)
        all_trials.append(np.full(n, t))
    return {
        "spike_times": np.concatenate(all_times),
        "trials": np.concatenate(all_trials),
        "rate": rate,
        "n_trials": n_trials,
        "trial_duration": duration,
    }


@pytest.fixture
def poisson_spikes():
    """Poisson 50 Hz spike train across 20 trials."""
    return make_poisson_spikes()


def make_identical_trials(
    n_spikes_per_trial: int = 10,
    duration: float = 2.5,
    n_trials: int = 20,
) -> dict:
    """Identical spike patterns across all trials.

    Trial-to-trial reliability should be perfect (r = 1).
    """
    template = np.linspace(0.5, 2.0, n_spikes_per_trial)
    return {
        "spike_times": np.tile(template, n_trials),
        "trials": np.repeat(np.arange(n_trials), n_spikes_per_trial),
        "rate": n_spikes_per_trial / duration,
        "n_trials": n_trials,
        "trial_duration": duration,
    }


@pytest.fixture
def identical_trials():
    """Identical spike patterns across 20 trials."""
    return make_identical_trials()


def make_globally_sorted_poisson(
    rate: float = 4.0,
    n_trials: int = 240,
    trial_dur: float = 2.5,
    rng_seed: int = SEED,
) -> dict:
    """Poisson spikes sorted globally by trial-relative time.

    This is the exact layout the example notebooks produce (after
    ``np.argsort(spike_times)``) and the regime where the
    trial-relative bug class bites hardest.
    """
    rng = _test_rng(rng_seed)
    spk: list[npt.NDArray] = []
    tr: list[npt.NDArray] = []
    for t in range(n_trials):
        n = rng.poisson(rate * trial_dur)
        spk.append(np.sort(rng.uniform(0, trial_dur, n)))
        tr.append(np.full(n, t, dtype=np.int64))
    spk_arr = np.concatenate(spk)
    tr_arr = np.concatenate(tr)
    order = np.argsort(spk_arr)
    return {
        "spike_times": spk_arr[order],
        "trials": tr_arr[order],
        "rate": rate,
        "n_trials": n_trials,
        "trial_duration": trial_dur,
    }


@pytest.fixture
def globally_sorted_poisson():
    """240-trial, 4 Hz Poisson spike train sorted globally by spike time."""
    return make_globally_sorted_poisson()


def make_bursting_spikes(
    n_trials: int = 100,
    trial_dur: float = 2.5,
    bursts_per_trial: int = 5,
    spikes_per_burst: int = 3,
    burst_isi: float = 0.002,
    rng_seed: int = SEED,
) -> dict:
    """Bursty spike train: clusters of tightly-spaced spikes.

    LV should be > 1.5 (bursty); the trial-relative bug used to
    mask this and report LV ≈ 1.
    """
    rng = _test_rng(rng_seed)
    spk: list[npt.NDArray] = []
    tr: list[npt.NDArray] = []
    for t in range(n_trials):
        base = rng.uniform(0, trial_dur - 0.01, bursts_per_trial)
        burst = np.sort(
            np.concatenate(
                [np.array([b + i * burst_isi for i in range(spikes_per_burst)]) for b in base]
            )
        )
        spk.append(burst)
        tr.append(np.full(len(burst), t, dtype=np.int64))
    spk_arr = np.concatenate(spk)
    tr_arr = np.concatenate(tr)
    order = np.argsort(spk_arr)
    return {
        "spike_times": spk_arr[order],
        "trials": tr_arr[order],
        "n_trials": n_trials,
        "trial_duration": trial_dur,
    }


@pytest.fixture
def bursting_spikes():
    """100-trial bursty spike train (5 bursts of 3 spikes per trial)."""
    return make_bursting_spikes()


# ==================================================================
# TUNING FIXTURES
# ==================================================================


# Re-export the canonical implementation from ``neural_cca.synthetic``.
# Keeping a thin shim here (rather than removing the conftest entry
# entirely) preserves the long-standing ``from tests.conftest import
# make_tuned_spikes`` import path used across the test suite.  The
# function is identical to the one previously inlined here — same
# RNG construction (``PCG64DXSM`` via ``SeedSequence``), same default
# seed (42), same per-trial Poisson recipe — so seeded fixtures keep
# producing bit-identical streams.
from neural_cca.synthetic import make_tuned_spikes  # noqa: E402


@pytest.fixture
def tuned_neuron():
    """Orientation-tuned neuron at 90° (12 directions × 20 repeats)."""
    st, tr, angles, unique_angles = make_tuned_spikes()
    return {
        "spike_times": st,
        "trials": tr,
        "angles": angles,
        "unique_angles": unique_angles,
        "preferred_angle": 90.0,
        "sigma_deg": 30.0,
        "n_angles": 12,
        "n_repeats": 20,
    }


def make_von_mises_response(
    orientations_deg: npt.NDArray,
    R0: float = 2.0,
    A: float = 10.0,
    kappa: float = 3.0,
    theta0_deg: float = 90.0,
) -> np.ndarray:
    """Von Mises tuning curve (deterministic, no noise)."""
    theta = np.deg2rad(orientations_deg)
    theta0 = np.deg2rad(theta0_deg)
    return R0 + A * np.exp(kappa * np.cos(2 * (theta - theta0)))


def make_double_gaussian_response(
    orientations_deg: npt.NDArray,
    A1: float = 10.0,
    A2: float = 5.0,
    sigma_deg: float = 25.0,
    theta0_deg: float = 45.0,
    baseline: float = 2.0,
) -> np.ndarray:
    """Double Gaussian tuning curve (deterministic, no noise)."""
    theta = np.deg2rad(orientations_deg)
    theta0 = np.deg2rad(theta0_deg)
    sigma = np.deg2rad(sigma_deg)
    d1 = np.arctan2(np.sin(theta - theta0), np.cos(theta - theta0))
    d2 = np.arctan2(
        np.sin(theta - (theta0 + np.pi)),
        np.cos(theta - (theta0 + np.pi)),
    )
    return (
        A1 * np.exp(-(d1**2) / (2 * sigma**2)) + A2 * np.exp(-(d2**2) / (2 * sigma**2)) + baseline
    )


def make_psth(
    f_stim: float,
    duration: float,
    bin_size: float,
    dc: float,
    f1_amp: float,
    f2_amp: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Synthetic PSTH with controlled harmonic content (deterministic)."""
    t = np.arange(0, duration, bin_size)
    psth = (
        dc + f1_amp * np.sin(2 * np.pi * f_stim * t) + f2_amp * np.sin(2 * np.pi * 2 * f_stim * t)
    )
    psth = np.maximum(psth, 0)  # firing rate can't be negative
    return psth, 1.0 / bin_size
