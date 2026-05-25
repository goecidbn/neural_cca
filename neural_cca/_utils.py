"""Private utility helpers shared across neural_cca submodules.

This is the single source of truth for cross-package helpers.  The
leading underscore signals that this module is package-private — public
re-exports live in :mod:`neural_cca` (top level) and the
relevant subpackage ``__init__`` modules.

Contents
--------
General-purpose:
    :func:`guarded_divide`  — element-wise divide that returns the
    numerator when the denominator is zero.
    :func:`steps2degree`    — N equally spaced angles covering 360°.

Circular statistics (degrees):
    :func:`circ_dist`  — smallest distance between two angles on a
    circle of arbitrary period.  Replaces the old
    ``circular_distance_deg``.
    :func:`wrap180`    — fold an angle into ``[0, 180)``.  Use for
    orientations.
    :func:`wrap360`    — fold an angle into ``[0, 360)``.  Use for
    directions.
    :func:`circ_mean`  — circular mean of angles, supporting both
    orientation (period 180°) and direction (period 360°), with
    optional weights.

Add new shared helpers here, not to a per-subpackage ``utils.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "guarded_divide",
    "make_rng",
    "steps2degree",
    "circ_dist",
    "wrap180",
    "wrap360",
    "circ_mean",
]


# ---------------------------------------------------------------------------
# RNG factory
# ---------------------------------------------------------------------------


def make_rng(
    seed: np.random.Generator | int | None = None,
) -> np.random.Generator:
    """Materialise a ``PCG64DXSM``-backed :class:`numpy.random.Generator`.

    ``PCG64DXSM`` (the "DXSM" output-function variant of the PCG64
    family) avoids the known parallel-stream self-correlation issues
    of the plain ``PCG64`` bit-generator that ``default_rng`` returns
    (numpy issue #16313).

    The *seed* argument is flexible:

    * ``Generator`` — returned as-is (the caller already owns it).
    * ``int`` — wrapped in a ``SeedSequence`` so even small integers
      get proper entropy scrambling.
    * ``None`` — fresh OS-entropy via ``SeedSequence()``.

    Every place in this package that needs a ``Generator`` calls this
    function instead of ``np.random.default_rng``.

    Args:
        seed: An existing Generator (passed through), an integer seed,
            or ``None`` for fresh OS entropy.

    Returns:
        A ``numpy.random.Generator`` backed by ``PCG64DXSM``.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    from numpy.random import PCG64DXSM, Generator, SeedSequence

    ss = SeedSequence(seed) if seed is not None else SeedSequence()
    return Generator(PCG64DXSM(ss))


# ---------------------------------------------------------------------------
# General-purpose helpers
# ---------------------------------------------------------------------------


def guarded_divide(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> np.ndarray | float:
    """Element-wise division that returns *x* where *y* is zero.

    Handles both scalar and array inputs.  Returns a Python float when
    both inputs are scalar, otherwise a numpy array.

    Args:
        x: Numerator (scalar or array).
        y: Denominator (scalar or array).

    Returns:
        Result of element-wise division with zero-denominators mapped
        to the corresponding numerator value.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    out = x_arr.copy()
    result = np.divide(x_arr, y_arr, out=out, where=(y_arr != 0))
    if result.ndim == 0:
        return result.item()
    return result


def steps2degree(n_steps: int) -> dict[int, float]:
    """Generate a mapping from 1-based step index to angle in degrees.

    Produces *n_steps* equally-spaced angles covering 360 degrees,
    starting at 0.

    Args:
        n_steps: Number of equidistant angular steps.

    Returns:
        Dictionary mapping ``{1: 0.0, 2: step, ..., n_steps: (n_steps-1)*step}``
        where ``step = 360 / n_steps``.

    Example:
        >>> steps2degree(12)
        {1: 0.0, 2: 30.0, 3: 60.0, ..., 12: 330.0}
    """
    step = 360.0 / n_steps
    return {i: (i - 1) * step for i in range(1, n_steps + 1)}


# ---------------------------------------------------------------------------
# Circular statistics (all angles in degrees)
# ---------------------------------------------------------------------------


def wrap360(angle_deg: npt.ArrayLike) -> np.ndarray | float:
    """Fold an angle (or array of angles) into ``[0, 360)``.

    Use this for *direction* angles where 0° and 360° denote the same
    direction.  Negative inputs and values above 360° are wrapped
    correctly.

    Args:
        angle_deg: Scalar or array of angles in degrees.

    Returns:
        The wrapped angle(s).  Scalar input → ``float``; array input →
        ``np.ndarray``.
    """
    arr = np.asarray(angle_deg, dtype=np.float64)
    result = arr % 360.0
    if result.ndim == 0:
        return float(result)
    return result


def wrap180(angle_deg: npt.ArrayLike) -> np.ndarray | float:
    """Fold an angle (or array of angles) into ``[0, 180)``.

    Use this for *orientation* angles where 0° and 180° denote the same
    orientation (e.g. a horizontal grating).  Negative inputs and values
    above 180° are wrapped correctly.

    Args:
        angle_deg: Scalar or array of angles in degrees.

    Returns:
        The wrapped angle(s).  Scalar input → ``float``; array input →
        ``np.ndarray``.
    """
    arr = np.asarray(angle_deg, dtype=np.float64)
    result = arr % 180.0
    if result.ndim == 0:
        return float(result)
    return result


def circ_dist(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    period: float = 360.0,
) -> np.ndarray | float:
    """Smallest distance between two angles on a circle of given *period*.

    Returns a value in ``[0, period/2]`` — the shorter arc-length
    between *a* and *b*, both in degrees.  Handles wraparound
    correctly, so e.g. ``circ_dist(350, 10) == 20``.

    Linear ``abs(a - b)`` is **wrong** for any code that uses
    ``argmin`` to find the nearest sampled angle to a target — when
    the target is close to the wraparound point, linear distance can
    pick a sample on the wrong side of the circle.  Use this helper
    instead of ``np.abs(angles - target)`` whenever the angles span the
    full circle.

    Args:
        a: Scalar or array of angles in degrees.
        b: Scalar or array of angles in degrees (broadcasts with *a*).
        period: Period of the circle in degrees.  Use ``360.0``
            (default) for directions and ``180.0`` for orientations.

    Returns:
        Distance value(s).  Scalar inputs → ``float``; array inputs →
        ``np.ndarray``.
    """
    diff = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) % period
    result = np.minimum(diff, period - diff)
    if result.ndim == 0:
        return float(result)
    return result


def circ_mean(
    angles_deg: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    period: float = 360.0,
) -> float:
    """Circular mean of angles in degrees.

    Computes the resultant-vector mean of *angles_deg*, optionally
    weighted by *weights*.  The *period* selects between direction
    (360°) and orientation (180°) statistics; for orientation, the
    angles are doubled before averaging and the result is halved
    after, which is the standard fix for the orientation/direction
    ambiguity.

    Args:
        angles_deg: Scalar or array of angles in degrees.
        weights: Optional non-negative weights with the same shape as
            *angles_deg* (e.g. firing rates at each orientation).
            ``None`` (default) → all weights equal to one.
        period: Period of the circle in degrees.  ``360.0`` for
            directions, ``180.0`` for orientations.

    Returns:
        Mean angle in degrees, wrapped into ``[0, period)``.  Returns
        ``np.nan`` when the resultant vector has zero magnitude (the
        circular mean is undefined in that case).
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    if weights is None:
        w = np.ones_like(angles)
    else:
        w = np.asarray(weights, dtype=np.float64)

    # Multiplier maps angles into the unit-circle representation.
    # k = 1 for direction (period = 360°), k = 2 for orientation
    # (period = 180°).  The doubled-angle trick collapses opposite
    # directions onto the same complex-plane point, so the mean of e.g.
    # 10° and 190° comes out at 10° rather than 100°.
    k = 360.0 / period
    theta = np.deg2rad(k * angles)
    vec = np.sum(w * np.exp(1j * theta))
    if np.abs(vec) == 0.0:
        return float("nan")
    mean_rad = float(np.angle(vec))
    result = (np.rad2deg(mean_rad) / k) % period
    # Floating-point quirk: a tiny negative `mean_rad` (e.g. -1e-17 from
    # np.angle on a vector with imag ≈ 0) becomes `period` after the
    # modulo because the float representation rounds up.  Wrap that
    # corner back to 0 so the contract "result < period" holds.
    if result >= period:
        result -= period
    return float(result)
