"""RNG-policy guards for the analysis leaf (``CROSS_CHECKS.md`` → RNG policy).

``make_rng()`` (PCG64DXSM) and ``_as_seed()`` are the **only** RNG
constructors allowed in shipping code. Banned in package source:

* ``np.random.default_rng`` — PCG64 parallel-stream self-correlation
  (numpy/numpy#16313);
* ``RandomState`` / ``MT19937`` / plain ``PCG64`` and the legacy
  ``np.random.<fn>`` global API.

These guards scan package **source only** — test fixtures may still use
``np.random.default_rng`` for brevity (package-only enforcement scope).
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

import neural_cca
from neural_cca.sorting.sorting import _as_seed

_SRC = pathlib.Path(neural_cca.__file__).resolve().parent

# Forbidden when accessed on the ``np.random`` module …
_NP_RANDOM_BANNED = frozenset(
    {
        "default_rng",
        "RandomState",
        "PCG64",
        "MT19937",
        "seed",
        "rand",
        "randn",
        "randint",
        "random_sample",
        "ranf",
        "sample",
        "choice",
        "permutation",
        "shuffle",
        "normal",
        "uniform",
        "standard_normal",
        "poisson",
        "binomial",
        "beta",
        "gamma",
    }
)
# … and when imported bare from ``numpy.random``.
_BARE_BANNED = frozenset({"default_rng", "RandomState", "PCG64", "MT19937"})


def _is_np_random(node: ast.AST) -> bool:
    """True if *node* is the ``np.random`` / ``numpy.random`` module reference."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "random"
        and isinstance(node.value, ast.Name)
        and node.value.id in {"np", "numpy"}
    )


def _violations(tree: ast.AST):
    """Yield ``(lineno, label)`` for each banned RNG construction call.

    Only flags ``np.random.<banned>(...)`` and bare ``default_rng(...)`` /
    ``RandomState(...)`` / ``PCG64(...)`` — never ``rng.<method>(...)``,
    so legitimate ``Generator`` methods (``rng.choice``, ``rng.integers``,
    …) and ``SeedSequence`` / ``PCG64DXSM`` / ``Generator`` are allowed.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _NP_RANDOM_BANNED
            and _is_np_random(func.value)
        ):
            yield node.lineno, f"np.random.{func.attr}"
        elif isinstance(func, ast.Name) and func.id in _BARE_BANNED:
            yield node.lineno, func.id


def test_no_banned_rng_constructors_in_package_source() -> None:
    """Package code must construct RNGs only via make_rng() / _as_seed()."""
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, what in _violations(tree):
            offenders.append(f"{path.relative_to(_SRC)}:{lineno} -> {what}()")
    assert not offenders, (
        "RNG policy violated — only make_rng()/_as_seed() may construct RNGs "
        "in package code (no default_rng / RandomState / legacy np.random / "
        f"plain PCG64): {offenders}"
    )


@pytest.mark.parametrize("seed", [0, 1, 42, 2**31, 2**32, 2**53, 2**128 - 1])
def test_as_seed_returns_uint32_and_is_deterministic(seed: int) -> None:
    """Any int — including a 128-bit master — yields a stable uint32."""
    s = _as_seed(seed)
    assert isinstance(s, int) and 0 <= s < 2**32
    assert _as_seed(seed) == s  # deterministic: same master → same sklearn seed


def test_as_seed_none_and_generator() -> None:
    from neural_cca import make_rng

    assert _as_seed(None) is None
    s = _as_seed(make_rng(0))
    assert isinstance(s, int) and 0 <= s < 2**32


def test_entropy_seed_roundtrips_through_pipeline() -> None:
    """C2 regression: a ~128-bit ``SeedSequence().entropy`` seed (what the
    bridge records in provenance) must run through the pipeline without
    sklearn rejecting it, and be reproducible. Before the ``_as_seed`` fix
    this raised ``InvalidParameterError``.
    """
    from numpy.random import SeedSequence

    from neural_cca import SortingData, make_rng, run_sorting_pipeline

    g = make_rng(7)
    half = 30
    waveforms = np.vstack(
        [g.standard_normal((half, 12)) + 4.0, g.standard_normal((half, 12)) - 4.0]
    )
    spike_times = g.uniform(0.6, 2.4, 2 * half)
    trials = np.repeat(np.arange(6), 10)
    angles = np.arange(0.0, 360.0, 60.0)
    data = SortingData(
        waveforms=waveforms,
        spike_times=spike_times,
        trials=trials,
        angles=angles,
        waveform_fs=30_000.0,
        n_trials=6,
        stim_window=(0.5, 2.5),
        stim_frequency=None,
    )

    seed = int(SeedSequence().entropy)  # ~128-bit, like load_from_visioniceio mints
    r1 = run_sorting_pipeline(data, rng=seed, plot=False, compute_os=False)
    r2 = run_sorting_pipeline(data, rng=seed, plot=False, compute_os=False)
    assert np.array_equal(r1.cluster_labels, r2.cluster_labels)
