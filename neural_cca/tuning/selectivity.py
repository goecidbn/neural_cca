r"""Orientation and direction selectivity indices.

Two families of metrics, distinguished by whether they use the **whole
tuning curve** (vector sum across all orientations) or only **two
points** (the preferred orientation and its orthogonal / null
counterpart).  The modern V1 literature (Mazurek, Kager & Van Hooser
2014) prefers the vector-sum family because it is robust to noise on
single bins.

**Vector-sum (global) family.**

- :func:`dosi_circular_normalised` — the primary implementation;
  computes :math:`|\sum_\theta R(\theta) e^{2i\theta}| / \sum R(\theta)`
  for orientation, single-angle for direction.
- :func:`circular_variance` — Ringach 2002 definition,
  ``CirVar = 1 - dosi_circular_normalised``.
- :func:`gosi` — **alias** for ``dosi_circular_normalised(...)`` (the
  *global* OSI in the modern convention; equal to ``1 - CirVar``).
- :func:`gdsi` — **alias** for the direction variant
  (``dosi_circular_normalised(..., direction_selectivity=True)``).

**Two-point family.**

- :func:`osi_two_point` — Niell & Stryker 2008 two-point ratio
  :math:`(R_\text{pref} - R_\text{orth}) / (R_\text{pref} + R_\text{orth})`,
  where the preferred orientation is found via the vector-sum (not
  the noisiest single bin).
- :func:`dsi_two_point` — direction analogue using
  :math:`R_\text{null} = R(\theta_\text{pref} + 180°)`.

The two-point ratios coincide with the vector-sum values only when
the tuning curve is well-described by a single cosine bump; in
general they differ, and reviewers will want both numbers if you
report either.  Reach for the vector-sum family by default.

References:
    Mazurek, M., Kager, M. & Van Hooser, S. D. (2014).  *Robust
    quantification of orientation selectivity and direction
    selectivity*.  Frontiers in Neural Circuits 8:92.
    doi:10.3389/fncir.2014.00092.

    Ringach, D. L., Shapley, R. M. & Hawken, M. J. (2002).
    *Orientation selectivity in macaque V1: diversity and laminar
    dependence*.  Journal of Neuroscience 22(13), 5639–5651.
    doi:10.1523/JNEUROSCI.22-13-05639.2002.

    Niell, C. M. & Stryker, M. P. (2008).  *Highly selective
    receptive fields in mouse visual cortex*.  Journal of
    Neuroscience 28(30), 7520–7536.
    doi:10.1523/JNEUROSCI.0623-08.2008.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._utils import circ_dist, circ_mean, wrap180, wrap360

__all__ = [
    "dosi_circular_normalised",
    "circular_variance",
    "gosi",
    "gdsi",
    "osi_two_point",
    "dsi_two_point",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rayleigh_test(angles_rad: npt.NDArray, weights: npt.NDArray) -> float:
    """Rayleigh test *p*-value for non-uniformity of weighted circular data.

    Uses the approximation for the Rayleigh *Z* statistic:

    .. math::
        Z = n \\cdot R^2,\\quad
        p \\approx e^{-Z}\\,(1 + (2Z - Z^2)/(4n) - (24Z - 132Z^2 + 76Z^3 - 9Z^4)/(288n^2))

    Reference: Mardia & Jupp (2000), *Directional Statistics*, §5.2.
    """
    n = float(np.sum(weights))
    if n == 0:
        return 1.0
    # Resultant length
    C = float(np.sum(weights * np.cos(angles_rad)))
    S = float(np.sum(weights * np.sin(angles_rad)))
    R = np.sqrt(C**2 + S**2) / n
    Z = n * R**2
    # Approximation from Mardia & Jupp (Eq. 5.2.5)
    p = np.exp(-Z) * (
        1.0 + (2 * Z - Z**2) / (4 * n) - (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2)
    )
    return max(0.0, min(1.0, float(p)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dosi_circular_normalised(
    activities: npt.NDArray[np.float64],
    angles: int | npt.NDArray[np.float64] | None = None,
    direction_selectivity: bool = False,
    return_unnormalised: bool = False,
    p_value: bool = False,
) -> float | np.complexfloating | dict:
    r"""Normalised orientation or direction selectivity index (OSI / DSI).

    .. math::

        \text{OSV} = \sum_\theta R(\theta)\,e^{2i\theta},\quad
        \text{OSI} = \frac{|\text{OSV}|}{\sum_\theta R(\theta)}

        \text{DSV} = \sum_\theta R(\theta)\,e^{1i\theta},\quad
        \text{DSI} = \frac{|\text{DSV}|}{\sum_\theta R(\theta)}

    OSI uses doubled angles (orientation space), DSI uses single angles
    (direction space).  Values range from 0 (no selectivity) to 1
    (perfect selectivity).

    Args:
        activities: Activity at each angle (e.g. mean firing rates).
        angles: Either an explicit angle array (degrees), or an
            ``int`` *N* — in which case the function builds
            ``np.linspace(0, 360, N, endpoint=False)``.  When given as
            an int, *N* **must** equal ``len(activities)``; otherwise
            the broadcast that follows would silently mis-align rates
            with angles.  ``None`` (the default) is equivalent to
            ``angles=len(activities)``.
        direction_selectivity: If ``True`` compute DSI; otherwise OSI.
        return_unnormalised: If ``True`` return the complex vector sum.
        p_value: If ``True`` return a dict ``{"value": float,
            "p_value": float}`` instead of a bare float.  The *p*-value
            is from a Rayleigh test for non-uniformity.

    Returns:
        Normalised index (float), unnormalised complex vector sum, or
        dict with ``value`` and ``p_value`` keys.

    Raises:
        ValueError: When *angles* is an ``int`` that does not match
            ``len(activities)``.
    """
    activities = np.asarray(activities, dtype=np.float64)

    # Default to equispaced angles matching the activities length so
    # the int-shorthand is never silently wrong.  The previous magic
    # default of ``8`` would error out (shape mismatch) for any other
    # activity length.
    if angles is None:
        angles = len(activities)
    if isinstance(angles, (int, np.integer)):
        n_ang = int(angles)
        if n_ang != len(activities):
            raise ValueError(
                "When `angles` is given as an integer it must equal "
                f"len(activities); got angles={n_ang}, "
                f"len(activities)={len(activities)}."
            )
        angles = np.linspace(0, 360, n_ang, endpoint=False)
    else:
        angles = np.asarray(angles, dtype=np.float64)

    angles_rad = np.deg2rad(angles)

    if direction_selectivity:
        vec = np.sum(activities * np.exp(1j * angles_rad))
    else:
        vec = np.sum(activities * np.exp(2j * angles_rad))

    if return_unnormalised:
        return vec

    value = float(np.abs(vec) / np.sum(activities))

    if p_value:
        test_angles = angles_rad if direction_selectivity else 2.0 * angles_rad
        pval = _rayleigh_test(test_angles, activities)
        return {"value": value, "p_value": pval}

    return value


def circular_variance(
    responses: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    p_value: bool = False,
) -> float | dict:
    r"""Circular variance of orientation tuning (1 - OSI).

    .. math::

        \text{CirVar} = 1 - \frac{|\sum R(\theta)\,e^{2i\theta}|}
                                   {\sum R(\theta)}

    Returns 0 for perfect selectivity, 1 for no selectivity.

    Args:
        responses: Mean rates at each angle.
        angles: Angles in degrees.
        p_value: If ``True`` return a dict ``{"value": float,
            "p_value": float}`` with a Rayleigh test *p*-value.

    Returns:
        Circular variance (float in [0, 1]), or dict when ``p_value=True``.
    """
    if p_value:
        osi_result = dosi_circular_normalised(responses, angles, p_value=True)
        return {"value": 1.0 - osi_result["value"], "p_value": osi_result["p_value"]}
    return 1.0 - dosi_circular_normalised(responses, angles)


def osi_two_point(
    responses: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    p_value: bool = False,
) -> float | dict:
    r"""Two-point orientation selectivity index (Niell & Stryker 2008).

    .. math::

        \text{OSI}_\text{2pt} = \frac{R_\text{pref} - R_\text{orth}}
                                       {R_\text{pref} + R_\text{orth}}

    where :math:`R_\text{pref}` is the response at the preferred
    orientation and :math:`R_\text{orth}` is the response at the
    orthogonal orientation (preferred + 90°).  The preferred
    orientation is found via the vector-sum (response-weighted
    circular mean), **not** the noisiest single bin (argmax).

    Note that "two-point" in the name is descriptive: only two
    samples of the tuning curve enter the ratio, making this metric
    sensitive to noise in those two specific bins.  For the global
    (whole-tuning-curve) OSI use :func:`gosi` /
    :func:`dosi_circular_normalised`.

    Args:
        responses: Mean firing rates at each orientation.
        angles: Stimulus angles in degrees.
        p_value: If ``True`` return ``{"value": float, "p_value": float}``.
            The reported *p_value* is a rate-weighted Rayleigh statistic
            that is **not** a calibrated tail probability for tuning
            significance — see
            :func:`~neural_cca.tuning.statistics.orientation_selectivity_significance`
            for the V1-literature-standard permutation test.

    Returns:
        OSI two-point (float in [-1, 1]) or dict when ``p_value=True``.

    References:
        Niell, C. M. & Stryker, M. P. (2008).  *Highly selective
        receptive fields in mouse visual cortex*.  Journal of
        Neuroscience 28(30), 7520–7536.
        doi:10.1523/JNEUROSCI.0623-08.2008.

        Mazurek, M., Kager, M. & Van Hooser, S. D. (2014).  *Robust
        quantification of orientation selectivity and direction
        selectivity*.  Frontiers in Neural Circuits 8:92.
        doi:10.3389/fncir.2014.00092.
    """
    responses = np.asarray(responses, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)

    # Preferred orientation via the response-weighted circular mean (the
    # vector-sum estimator).  Earlier versions used ``argmax(responses)``
    # — winner-take-all on the noisiest single bin — which made the
    # ratio depend on whichever orientation happened to fire most rather
    # than on the underlying tuning curve.  Vector-sum is the convention
    # already used by ``preferred_dori`` and is robust to noise.
    pref_angle = circ_mean(angles, weights=responses, period=180.0)
    if np.isnan(pref_angle):
        # Silent neuron (zero resultant vector) — undefined.
        if p_value:
            angles_rad = np.deg2rad(angles)
            return {
                "value": np.nan,
                "p_value": _rayleigh_test(2.0 * angles_rad, responses),
            }
        return np.nan
    pref_idx = int(np.argmin(circ_dist(angles, pref_angle, period=180.0)))
    r_pref = float(responses[pref_idx])

    # Orthogonal = pref + 90° in orientation space.  ``+90`` and
    # ``−90`` from the preferred orientation fold to the *same* angle
    # modulo 180° (``wrap180(p+90) == wrap180(p-90)``), so a single
    # lookup suffices.  ``period=180`` is essential here: the default
    # 360° period would treat orientation wraparound as if 0° and 180°
    # were distinct, picking the wrong sample for any preferred
    # orientation near the 0°/180° seam.
    orth_angle = wrap180(pref_angle + 90)
    orth_idx = int(np.argmin(circ_dist(angles, orth_angle, period=180.0)))
    r_orth = float(responses[orth_idx])

    denom = r_pref + r_orth
    # NaN (not 0.0) when both pref and orth responses are zero — a
    # silent neuron has *undefined* selectivity, not "no selectivity".
    value = float((r_pref - r_orth) / denom) if denom > 0 else np.nan

    if p_value:
        angles_rad = np.deg2rad(angles)
        pval = _rayleigh_test(2.0 * angles_rad, responses)
        return {"value": value, "p_value": pval}
    return value


def dsi_two_point(
    responses: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    p_value: bool = False,
) -> float | dict:
    r"""Two-point direction selectivity index (Niell & Stryker 2008).

    .. math::

        \text{DSI}_\text{2pt} = \frac{R_\text{pref} - R_\text{null}}
                                       {R_\text{pref} + R_\text{null}}

    where :math:`R_\text{null}` is the response at the preferred
    direction + 180°.  The preferred direction is found via the
    vector-sum over 360° (not argmax).

    For the global (vector-sum, whole-tuning-curve) direction
    selectivity use :func:`gdsi` /
    :func:`dosi_circular_normalised(..., direction_selectivity=True)`.

    Args:
        responses: Mean firing rates at each direction.
        angles: Stimulus directions in degrees.
        p_value: See :func:`osi_two_point` — *p_value* is a
            rate-weighted Rayleigh and is **not** a calibrated tail
            probability for direction-tuning significance.

    Returns:
        DSI two-point (float in [-1, 1]) or dict when ``p_value=True``.

    References:
        Niell, C. M. & Stryker, M. P. (2008).  *Highly selective
        receptive fields in mouse visual cortex*.  Journal of
        Neuroscience 28(30), 7520–7536.
        doi:10.1523/JNEUROSCI.0623-08.2008.

        Mazurek, M., Kager, M. & Van Hooser, S. D. (2014).  *Robust
        quantification of orientation selectivity and direction
        selectivity*.  Frontiers in Neural Circuits 8:92.
        doi:10.3389/fncir.2014.00092.
    """
    responses = np.asarray(responses, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)

    # Preferred direction via the response-weighted circular mean
    # (vector sum) over the full 360° circle.
    pref_angle = circ_mean(angles, weights=responses, period=360.0)
    if np.isnan(pref_angle):
        # Silent neuron — undefined.
        if p_value:
            angles_rad = np.deg2rad(angles)
            return {
                "value": np.nan,
                "p_value": _rayleigh_test(angles_rad, responses),
            }
        return np.nan
    pref_idx = int(np.argmin(circ_dist(angles, pref_angle, period=360.0)))
    r_pref = float(responses[pref_idx])

    # Null = preferred + 180°.  Use circular distance with the full
    # 360° period (direction space) to handle the 0°/360° wraparound;
    # linear distance would pick the wrong sample whenever the null
    # target sits across the seam.
    null_angle = wrap360(pref_angle + 180)
    null_idx = int(np.argmin(circ_dist(angles, null_angle, period=360.0)))
    r_null = responses[null_idx]

    denom = r_pref + r_null
    # NaN (not 0.0) when both pref and null responses are zero — silent
    # neurons have undefined selectivity.
    value = float((r_pref - r_null) / denom) if denom > 0 else np.nan

    if p_value:
        angles_rad = np.deg2rad(angles)
        pval = _rayleigh_test(angles_rad, responses)
        return {"value": value, "p_value": pval}
    return value


# ---------------------------------------------------------------------------
# Modern global-OSI / gDSI aliases.  Defined *after* osi_two_point /
# dsi_two_point so docstrings can cross-reference both.
# ---------------------------------------------------------------------------


def gosi(
    responses: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    p_value: bool = False,
) -> float | dict:
    r"""Global orientation selectivity index (vector-sum / 1 − CirVar).

    .. math::

        \text{gOSI} = \frac{\bigl|\sum_\theta R(\theta)\,
            e^{2i\theta}\bigr|}{\sum_\theta R(\theta)} = 1 - \text{CirVar}

    This is the **modern V1-literature standard** for "OSI"
    (Mazurek, Kager & Van Hooser 2014; Ringach, Shapley & Hawken
    2002): a global measure using *all* tuning-curve samples,
    robust to single-bin noise.  Use this by default.

    For the two-point ratio :math:`(R_\text{pref} - R_\text{orth}) /
    (R_\text{pref} + R_\text{orth})` (Niell & Stryker 2008) see
    :func:`osi_two_point`.

    This function is a thin alias for
    :func:`dosi_circular_normalised` with the orientation default —
    the implementations cannot drift apart.

    Args:
        responses: Mean firing rates at each orientation.
        angles: Stimulus angles in degrees.
        p_value: See :func:`dosi_circular_normalised`.  Note that the
            reported *p_value* is descriptive, not a calibrated tail
            probability — use
            :func:`~neural_cca.tuning.statistics.orientation_selectivity_significance`
            for permutation-test significance.

    Returns:
        gOSI (float in ``[0, 1]``) or dict when ``p_value=True``.

    References:
        Mazurek, M., Kager, M. & Van Hooser, S. D. (2014).  *Robust
        quantification of orientation selectivity and direction
        selectivity*.  Frontiers in Neural Circuits 8:92.
        doi:10.3389/fncir.2014.00092.

        Ringach, D. L., Shapley, R. M. & Hawken, M. J. (2002).
        *Orientation selectivity in macaque V1: diversity and
        laminar dependence*.  Journal of Neuroscience 22(13),
        5639–5651.  doi:10.1523/JNEUROSCI.22-13-05639.2002.
    """
    return dosi_circular_normalised(responses, angles, p_value=p_value)


def gdsi(
    responses: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    p_value: bool = False,
) -> float | dict:
    r"""Global direction selectivity index (vector-sum over 360°).

    .. math::

        \text{gDSI} = \frac{\bigl|\sum_\theta R(\theta)\,
            e^{i\theta}\bigr|}{\sum_\theta R(\theta)}

    Whole-tuning-curve direction selectivity (the modern convention,
    Mazurek 2014).  For the two-point ratio
    :math:`(R_\text{pref} - R_\text{null}) / (R_\text{pref} + R_\text{null})`
    see :func:`dsi_two_point`.

    Thin alias for ``dosi_circular_normalised(responses, angles,
    direction_selectivity=True, p_value=...)``.

    Args:
        responses: Mean firing rates at each direction.
        angles: Stimulus directions in degrees.
        p_value: See :func:`dosi_circular_normalised`.

    Returns:
        gDSI (float in ``[0, 1]``) or dict when ``p_value=True``.

    References:
        Mazurek, M., Kager, M. & Van Hooser, S. D. (2014).  *Robust
        quantification of orientation selectivity and direction
        selectivity*.  Frontiers in Neural Circuits 8:92.
        doi:10.3389/fncir.2014.00092.
    """
    return dosi_circular_normalised(
        responses,
        angles,
        direction_selectivity=True,
        p_value=p_value,
    )
