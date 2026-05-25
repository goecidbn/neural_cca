tuning — Orientation & Direction Selectivity
============================================

Overview
--------

The ``tuning`` subpackage implements the standard catalogue of
orientation- and direction-selectivity analyses used in visual
neuroscience: vector-sum selectivity indices, tuning-curve fitting,
F1 / F0 harmonic decomposition, temporal-frequency tuning,
population statistics, and statistical-significance testing.

All analyses share a single per-trial post-stimulus filter
(``_TrialFilteredSpikes``).  Building it walks the raw spike arrays
once; downstream consumers reuse the filtered ``(mfrs, angles,
spike_times_by_trial)`` triple via a private ``_filter=`` kwarg.
The composite :func:`get_os_metrics` builds the filter once at the
top of the call and forwards it to every helper that would
otherwise rebuild it — the regression test pins the build at
exactly one call regardless of which downstream metrics are
requested.

Scientific scope
----------------

**Selectivity indices.**  Two complementary families
(Mazurek, Kager & Van Hooser 2014, doi:10.3389/fncir.2014.00092):

* *Vector-sum* (the modern V1 default) —
  :func:`dosi_circular_normalised` returns
  :math:`\mathrm{OSI} = |\sum_\theta R(\theta) e^{2i\theta}| / \sum_\theta R(\theta)`
  on doubled angles (orientation space) and the analogous DSI on
  single angles (direction space).  :func:`gosi` and :func:`gdsi`
  are thin aliases for these so methods sections can reach for
  the name they expect.  :func:`circular_variance` returns
  :math:`1 - \mathrm{OSI}` (Ringach et al. 2002).  ``p_value=True``
  returns a rate-weighted Rayleigh statistic (Mardia & Jupp
  approximation, 5th-order) for **descriptive** use only — it is
  not a calibrated tail probability for tuning-curve significance.
  For a calibrated permutation test use
  :func:`orientation_selectivity_significance`.
* *Two-point* — :func:`osi_two_point` and :func:`dsi_two_point`
  return the Niell & Stryker (2008) ratios
  :math:`(R_\mathrm{pref} - R_\mathrm{orth}) / (R_\mathrm{pref} + R_\mathrm{orth})`
  with the preferred orientation estimated by the vector-sum
  circular mean (not winner-take-all argmax).  Orthogonal lookups
  use ``period=180`` / ``period=360`` so cells with a preferred
  orientation near the 0° / 180° (or 0° / 360°) seam pick the
  correct sample.  These functions were named ``gosi`` / ``gdsi``
  before v0.2.0 — see the changelog for the rename.

**Curve fitting.**  :func:`von_mises_fit` accepts
``tuning_type="orientation"`` (single bump on the half-circle,
:math:`R_0 + A \exp(\kappa \cos(2(\theta - \theta_0)))`) or
``"direction"`` (two bumps at the preferred and null directions).
Returns a uniform dict ``{preferred_angle, kappa, baseline,
r_squared, bandwidth_hwhh, ...}``; the bandwidth is the
half-width-at-half-height on the underlying angle axis (the
orientation form's doubled-angle structure is handled
internally).  :func:`double_gaussian_fit` provides an alternative
two-bump parameterisation in degree space.

:func:`tuning_curve_interpolation` returns the preferred angle from
the *peak of the fitted curve* sampled across one full period
(180° for orientation, 360° for direction / double Gaussian) —
sampling only the observed angle range would miss the wraparound
and return an angle inside ``[angles.min(), angles.max()]`` for a
cell whose true peak sits at e.g. 350°.

**Harmonic analysis.**  :func:`compute_f0_f1_f2` extracts the DC,
F1, and F2 harmonic amplitudes from per-trial PSTHs via FFT at
``f_stim``, ``2 * f_stim``.  Works on 1-D and batched PSTHs.
:func:`f1_phase` returns the F1 phase in radians, used by
trial-to-trial phase-consistency analyses.
:func:`modulation_ratio_per_orientation` reports the per-angle F1/F0
ratio (simple cells: > 1 at preferred; complex cells: < 1 across
all angles).  :func:`cross_orientation_suppression` returns a proxy
suppression index :math:`1 - R_\mathrm{orth} / R_\mathrm{pref}`
computed from the tuning curve.

**Temporal frequency.**  :func:`temporal_frequency_tuning` builds a
TF tuning curve from F1 amplitude or mean rate at each TF, fits a
log-Gaussian, and reports preferred TF, bandwidth (octaves), and
R².

**Population analyses.**
:func:`orientation_map_statistics` returns the circular mean and
concentration of a population's preferred-orientation distribution
with a Rayleigh test for non-uniformity.
:func:`signal_correlations` and :func:`noise_correlations` return
pairwise (n, n) Pearson matrices on tuning curves and on z-scored
trial residuals respectively.

**Statistical testing.**
:func:`orientation_selectivity_significance` runs the V1-literature
standard permutation null (shuffle responses, keep angles fixed —
intentionally; this is the right null for "rates depend on
angles") with the Phipson & Smyth (2010) ``+1`` correction so the
smallest reportable :math:`p` is :math:`1/(B+1)` instead of ``0``.
A rate-weighted Rayleigh statistic is also returned for legacy
reporting but is documented as descriptive only.
:func:`anova_across_orientations` runs a one-way F-test on trial
firing rates across angles.
:func:`bootstrap_ci_strata` returns stratified-bootstrap CIs that
preserve the (rate, angle) pairing — what the composite
:func:`get_os_metrics` uses for its OSI / DSI / gOSI / gDSI CIs.
The default ``method="bca"`` (Efron 1987, bias-corrected and
accelerated) is second-order accurate and transformation-respecting
— the right choice for boundary-bounded statistics like OSI in
``[0, 1]``.  ``method="percentile"`` is available for backwards
compatibility.  Plain :func:`bootstrap_ci` is provided for
non-paired statistics (e.g. mean of a flat sample).

**Composite.**  :func:`get_os_metrics` is the all-in-one entry
point used by :func:`run_sorting_pipeline`'s per-cluster OS
evaluation.  It returns an :class:`OsMetricsResult` ``TypedDict``
with three verbosity tiers (``return_verbose=0/1/2``) and optional
gOSI/gDSI, Rayleigh + ANOVA p-values, curve fitting, and bootstrap
CIs gated by individual flags.

API reference
-------------

.. automodule:: neural_cca.tuning
   :no-members:

Selectivity
~~~~~~~~~~~

.. automodule:: neural_cca.tuning.selectivity
   :members:

Tuning
~~~~~~

.. automodule:: neural_cca.tuning.tuning
   :members:

Fitting
~~~~~~~

.. automodule:: neural_cca.tuning.fitting
   :members:

Modulation
~~~~~~~~~~

.. automodule:: neural_cca.tuning.modulation
   :members:

Temporal
~~~~~~~~

.. automodule:: neural_cca.tuning.temporal
   :members:

Population
~~~~~~~~~~

.. automodule:: neural_cca.tuning.population
   :members:

Statistics
~~~~~~~~~~

.. automodule:: neural_cca.tuning.statistics
   :members:

Plotting
~~~~~~~~

.. automodule:: neural_cca.tuning.plotting
   :members:
