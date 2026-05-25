sorting
=======

Overview
--------

The ``sorting`` subpackage clusters extracellular spike waveforms
into putative single units and reports a battery of quality metrics
on the result.  The pipeline is intentionally minimal: per-feature
*z*-scoring, principal-component projection, k-means clustering, and
silhouette-based selection of the cluster count.  It is the
methods-section pipeline that gets cited in published papers — not
a deep network or template-matching engine — so the algorithmic
choices are short to describe and easy to reproduce.

Scientific scope
----------------

**Clustering.**  Waveform snippets ``(n_spikes, snippet_length)`` are
*z*-scored per feature (so a single high-amplitude sample cannot
dominate the principal axes) and projected onto the top principal
components that retain 95 % of the variance.  k-means is then run on
the projected scores; the silhouette score is evaluated **in that
same feature space** and used to pick the cluster count *k* from a
user-supplied range.  Reporting silhouette in raw waveform space
while clustering happens in PCA space — the previous behaviour — is
internally inconsistent and is no longer the default.

**Quality metrics.**  Each cluster is scored along three orthogonal
axes:

* *Feature-space separation* — silhouette score,
  :func:`isolation_distance` (Harris et al. 2000),
  :func:`l_ratio` (Schmitzer-Torbert et al. 2005), and
  :func:`d_prime` (signal-detection-theory cluster separation).
  Cluster covariances are estimated with the **Ledoit–Wolf
  shrinkage estimator**; see :func:`_ledoit_wolf_precision` for the
  rationale.  All four metrics live in the same feature space the
  clustering used.  ``isolation_distance`` and ``l_ratio`` accept
  ``mode="worst_pair"`` (Sibille et al. 2024) to report the
  worst-neighbour value instead of pooling all non-cluster spikes —
  the modern best-practice convention used by the Allen Institute
  ``ecephys_spike_sorting`` pipeline.  Default is the original
  ``mode="global"`` (Harris / Schmitzer-Torbert).
* *Amplitude / shape* — :func:`est_snr` (peak-to-peak vs. residual
  std), :func:`peak_amplitude_snr` (peak vs. baseline std),
  :func:`waveform_stability` (early-vs-late mean-waveform
  Pearson *r*), :func:`amplitude_drift` (Spearman of peak amplitude
  vs. spike index), and :func:`fraction_missing` (tail estimate of
  spikes lost below the detection threshold).  ``fraction_missing``
  ships three methods: ``method="gaussian"`` (default; Hill 2011 /
  Allen Institute convention, the universally reported number),
  ``method="lognormal"`` (better-calibrated for the typical V1
  lognormal amplitude distribution; Buzsáki & Mizuseki 2014), and
  ``method="empirical"`` (non-parametric Gaussian-KDE tail
  extrapolation; Silverman 1986).  These operate on the *raw*
  waveforms because voltage amplitude is only defined there.
* *Refractory contract* — :func:`rpvs` reports the
  SpikeInterface-style "violations per spike" rate, while
  :func:`contamination_rate_hill` returns the calibrated Hill et
  al. (2011) contamination *fraction* :math:`C \in [0, 0.5]` — the
  metric reviewers expect when methods sections quote
  "contamination < X %".  Both are scored at the
  ``refractory_period`` default of 1 ms with the strict ``<`` rule
  (a pair at exactly the refractory period is not a violation).
  ``contamination_rate_hill`` is wired into
  :func:`evaluate_sorting` so the standard quality dict carries it
  alongside ``abs_rpvs`` / ``rel_rpvs``.

**Single-cluster mode (k=1).**  ``run_sorting_pipeline(n_clusters=1)``
is a first-class supported path for pre-isolated single-unit
channels (Kilosort export, manual curation, low-density "trust the
channel" recordings).  At k=1 the silhouette-class metrics are
mathematically undefined and the pipeline NaN-fills them with a
single :class:`RuntimeWarning`; everything else stays numeric.
Setting ``min_silhouette`` on auto-select gives a soft fallback to
k=1 whenever no candidate clears the threshold.

**Zarr export.**  Two layouts are provided — *flat* keeps the
original ``(n_spikes, ...)`` shape, *clustered* pads to
``(n_clusters, max_spikes, ...)`` so each cluster occupies one
chunk.  The clustered layout records an ``original_index`` array so
the round-trip is identity-preserving on every per-spike array; a
``QualityMetricKind`` registry tags each entry so the writer can
dispatch on shape without read-time inference.

API reference
-------------

.. automodule:: neural_cca.sorting
   :no-members:

.. automodule:: neural_cca.sorting.containers
   :members:

.. automodule:: neural_cca.sorting.io_util
   :members:
   :exclude-members: SortingData, SortingResult

.. automodule:: neural_cca.sorting.sorting
   :members:

.. automodule:: neural_cca.sorting.batch
   :members:

.. automodule:: neural_cca.sorting.metrics
   :members:

.. automodule:: neural_cca.sorting.plotting
   :members:
