spike_train — Spike Train Statistics
====================================

Overview
--------

The ``spike_train`` subpackage computes per-unit spike-train
statistics on the *pre-sorted* spike times that come out of the
``sorting`` pipeline (or any external sorter).  It does *not*
re-detect spikes from raw voltage traces — every function takes
``(spike_times, trials, ...)`` arrays and returns scalar or
time-series statistics.

.. note::

   This subpackage was called ``neural_cca.sta`` before v0.2.0.  The
   old import path is still available as a deprecation shim that
   re-exports everything from ``spike_train`` and emits a
   :class:`DeprecationWarning` on first import.  The rename frees
   the ``sta`` name for the canonical *spike-triggered average*
   (Schwartz et al. 2006) meaning of "STA" in computational
   neuroscience.

The package convention is **trial-relative seconds**: spike times
are taken to restart at zero on every trial, and a per-spike
``trials`` array maps each spike to its trial index.  Every
ISI-based function (CV, LV, LvR, CV-log-ISI, autocorrelogram, ISI
violation rate) routes through a single trial-aware ISI helper
(:func:`_per_trial_isis`) so cross-trial spike pairs are never
silently treated as real ISIs.  Without this, a globally-sorted
spike train from a trial-based recording reports a 4 Hz Poisson
cell at ~600 Hz ISI-violation rate (cross-trial pairs whose
timestamps happen to fall within 1 ms dominate the count).

Scientific scope
----------------

**Rate.**  Mean firing rate (:func:`calc_mfr_trial`), peri-stimulus
time histogram (:func:`psth`), and a windowed firing-rate
stability metric (:func:`firing_rate_stability`).  The PSTH and
windowed metrics use a stable ``ceil + linspace`` bin construction
so the bin count is independent of float rounding on
``trial_duration / bin_size``.

**Variability.**  Coefficient of variation (CV), local variation
(LV) and its refractory-corrected variant (LvR — Shinomoto et al.
2009), CV of :math:`\log_{10}(\mathrm{ISI})`, and Fano factor.
All ISI-based metrics pool *within-trial* contributions across
trials — cross-trial pairs never enter the numerator.
``fano_factor`` distinguishes ``mode="per_trial"`` (the standard
neuroscience convention: each trial is one observation) from
``mode="per_bin"`` (bin-count Fano, sensitive to bin width); the
implicit-mode path is deprecated.

**Temporal structure.**  :func:`autocorrelogram` returns
within-trial pairwise spike-time differences.  The implementation
vectorises pair accumulation via ``np.searchsorted`` plus two
batched ``np.histogram`` calls per spike train (no per-pair
histogram cost).  Pass ``normalize="rate"`` to convert raw counts
to the per-central-spike rate convention used by
elephant / SpikeInterface; ``"counts"`` (default) returns the
integer pair counts.  :func:`isi_violation_rate` reports the
refractory-violation rate either as Hz across the total recording
duration or as a percentage of valid ISIs.

**Latency.**  :func:`first_spike_latency` reports per-trial
first-spike latencies after stimulus onset, plus mean / median /
std / responsive-fraction summaries.  The companion
:func:`first_spike_latency_thresholded` adds a baseline-relative
detection threshold and a response-window cap so spontaneous
spikes don't contaminate the "evoked latency" estimate (Reich et
al. 2000; Smyth et al. 2003).

**Trial reliability.**  :func:`trial_to_trial_reliability` returns
the mean Pearson correlation of each per-trial PSTH against the
across-trial mean PSTH — the standard visual-cortex "reliability
index", :math:`O(n_\mathrm{trials})` rather than the older
all-pairs :math:`O(n_\mathrm{trials}^2)` form.  An ``f1_phase`` mode
returns the mean resultant length of per-trial F1 phases (high
means phase-consistent).  :func:`trial_to_trial_correlation_matrix`
exposes the full pairwise PSTH correlation matrix for inspecting
block-wise drift or sub-populations of trials.

Plotting helpers (:func:`plot_isi_histogram`, :func:`plot_autocorrelogram`,
:func:`plot_psth`, etc.) follow the package-wide ``ax=None`` pattern
and forward the trial-aware path correctly.  ``plot_autocorrelogram``
emits a :class:`RuntimeWarning` when ``refractory_period`` is not a
whole multiple of ``bin_size`` (the dashed refractory line would
fall inside a bar rather than on a bin edge).

API reference
-------------

.. automodule:: neural_cca.spike_train
   :no-members:

.. automodule:: neural_cca.spike_train.analysis
   :members:

.. automodule:: neural_cca.spike_train.plotting
   :members:
