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
  :func:`l_ratio`, and :func:`d_prime` (signal-detection-theory
  cluster separation).  Cluster covariances are estimated with the
  **Ledoit–Wolf shrinkage estimator**; see
  :func:`_ledoit_wolf_precision` for the rationale.  All four
  metrics live in the same feature space the clustering used.
* *Amplitude / shape* — :func:`est_snr` (peak-to-peak vs. residual
  std), :func:`peak_amplitude_snr` (peak vs. baseline std),
  :func:`waveform_stability` (early-vs-late mean-waveform
  Pearson *r*), :func:`amplitude_drift` (Spearman of peak amplitude
  vs. spike index), and :func:`fraction_missing` (Gaussian-tail
  estimate of spikes lost below the detection threshold).  These
  operate on the *raw* waveforms because voltage amplitude is only
  defined there.
* *Refractory contract* — :func:`rpvs` counts inter-spike intervals
  shorter than ``refractory_period`` (default 1 ms).  The
  package-wide convention is the strict ``<`` rule: a pair separated
  by *exactly* the refractory period is not a violation.

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
