"""Population-level orientation analyses.

Provides orientation map statistics (Rayleigh test), and pairwise
signal and noise correlation matrices.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr

from .._utils import circ_mean
from .selectivity import _rayleigh_test

__all__ = [
    "orientation_map_statistics",
    "signal_correlations",
    "noise_correlations",
]


def orientation_map_statistics(
    preferred_orientations: npt.NDArray[np.float64],
) -> dict:
    """Statistics of a population's preferred-orientation distribution.

    Uses the Rayleigh test on doubled angles (orientation space) to
    test for non-uniformity.

    Args:
        preferred_orientations: Preferred orientations in degrees, one
            per neuron.

    Returns:
        Dict with keys:

        - ``"mean_ori"`` — circular mean orientation (degrees, 0–180)
        - ``"concentration"`` — resultant vector length (0 = uniform, 1 = identical)
        - ``"rayleigh_z"`` — Rayleigh Z statistic
        - ``"rayleigh_p"`` — Rayleigh p-value
        - ``"is_uniform"`` — ``True`` if p ≥ 0.05 (fail to reject uniformity)

    References:
        Mardia, K. V. & Jupp, P. E. (2000).  *Directional Statistics*,
        2nd ed., §5.2.  Wiley, Chichester.
        (The Rayleigh-test reference used here; the asymptotic
        expansion is Eq. 5.2.5.)

        Berens, P. (2009).  *CircStat: a MATLAB toolbox for circular
        statistics*.  Journal of Statistical Software 31(10).
        doi:10.18637/jss.v031.i10.
    """
    oris = np.asarray(preferred_orientations, dtype=np.float64)
    n = len(oris)
    if n == 0:
        return {
            "mean_ori": np.nan,
            "concentration": np.nan,
            "rayleigh_z": np.nan,
            "rayleigh_p": np.nan,
            "is_uniform": True,
        }

    # Double angles to map orientations (0-180) into full circle.
    # circ_mean handles the doubling internally; we keep `theta` for the
    # resultant length and Rayleigh test below.
    theta = np.deg2rad(2.0 * oris)
    weights = np.ones(n)

    mean_ori = circ_mean(oris, period=180.0)

    # Resultant length
    C = np.sum(np.cos(theta))
    S = np.sum(np.sin(theta))
    R = np.sqrt(C**2 + S**2) / n
    concentration = float(R)

    # Rayleigh test
    Z = n * R**2
    p = _rayleigh_test(theta, weights)

    return {
        "mean_ori": mean_ori,
        "concentration": concentration,
        "rayleigh_z": float(Z),
        "rayleigh_p": p,
        "is_uniform": p >= 0.05,
    }


def signal_correlations(
    tuning_curves: npt.NDArray[np.float64],
) -> np.ndarray:
    r"""Pairwise signal correlations between neurons.

    Signal correlations measure the similarity of tuning curves
    (mean response profiles across orientations).  The standard
    formulation is the Pearson correlation of the across-trial mean
    response vectors, as in Cohen & Kohn (2011) and Averbeck, Latham
    & Pouget (2006).

    Args:
        tuning_curves: Array of shape ``(n_neurons, n_orientations)``
            where each row is the mean response at each orientation.

    Returns:
        Symmetric ``(n_neurons, n_neurons)`` Pearson correlation matrix.
        Off-diagonal entries are ``np.nan`` whenever either neuron has a
        zero-variance (flat) tuning curve — Pearson's r is undefined in
        that case, matching the package-wide convention that ``np.nan``
        signals "undefined" rather than "uncorrelated".

    References:
        Cohen, M. R. & Kohn, A. (2011).  *Measuring and interpreting
        neuronal correlations*.  Nature Neuroscience 14(7), 811–819.
        doi:10.1038/nn.2842.

        Averbeck, B. B., Latham, P. E. & Pouget, A. (2006).  *Neural
        correlations, population coding and computation*.  Nature
        Reviews Neuroscience 7(5), 358–366.  doi:10.1038/nrn1888.
    """
    tuning_curves = np.asarray(tuning_curves, dtype=np.float64)
    n = tuning_curves.shape[0]
    corr = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            if np.std(tuning_curves[i]) == 0 or np.std(tuning_curves[j]) == 0:
                r = np.nan
            else:
                r, _ = pearsonr(tuning_curves[i], tuning_curves[j])
            corr[i, j] = r
            corr[j, i] = r

    return corr


def noise_correlations(
    trial_rates: npt.NDArray[np.float64],
    trial_angles: npt.NDArray[np.float64],
) -> np.ndarray:
    """Pairwise noise correlations between neurons.

    Noise correlations measure correlated trial-to-trial variability
    after subtracting each neuron's mean response per orientation
    (z-scored residuals).

    Args:
        trial_rates: Array of shape ``(n_neurons, n_trials)`` —
            firing rate of each neuron on each trial.
        trial_angles: Array of shape ``(n_trials,)`` — stimulus
            angle on each trial (degrees).

    Returns:
        Symmetric ``(n_neurons, n_neurons)`` Pearson correlation matrix
        of z-scored residuals.  Off-diagonal entries are ``np.nan``
        whenever either neuron has zero-variance residuals (e.g. an
        identical response on every repeat of every orientation) —
        Pearson's *r* is undefined in that case, matching the
        package-wide convention that ``np.nan`` signals "undefined"
        rather than "uncorrelated".

    References:
        Cohen, M. R. & Kohn, A. (2011).  *Measuring and interpreting
        neuronal correlations*.  Nature Neuroscience 14(7), 811–819.
        doi:10.1038/nn.2842.

        Bair, W., Zohary, E. & Newsome, W. T. (2001).  *Correlated
        firing in macaque visual area MT: time scales and
        relationship to behavior*.  Journal of Neuroscience 21(5),
        1676–1697.  doi:10.1523/JNEUROSCI.21-05-01676.2001.
    """
    trial_rates = np.asarray(trial_rates, dtype=np.float64)
    trial_angles = np.asarray(trial_angles, dtype=np.float64)
    n_neurons, n_trials = trial_rates.shape
    unique_angles = np.unique(trial_angles)

    # Compute z-scored residuals per neuron per orientation
    residuals = np.zeros_like(trial_rates)
    for ang in unique_angles:
        mask = trial_angles == ang
        for i in range(n_neurons):
            rates_at_ang = trial_rates[i, mask]
            mean_rate = np.mean(rates_at_ang)
            std_rate = np.std(rates_at_ang)
            if std_rate > 0:
                residuals[i, mask] = (rates_at_ang - mean_rate) / std_rate
            else:
                residuals[i, mask] = 0.0

    # Pairwise correlations of residuals.  Off-diagonal NaN signals
    # "undefined" (e.g. a neuron with identical trial responses at every
    # orientation produces zero-variance residuals) — consistent with
    # the package-wide undefined-vs-zero convention.
    corr = np.eye(n_neurons)
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if np.std(residuals[i]) == 0 or np.std(residuals[j]) == 0:
                r = np.nan
            else:
                r, _ = pearsonr(residuals[i], residuals[j])
            corr[i, j] = r
            corr[j, i] = r

    return corr
