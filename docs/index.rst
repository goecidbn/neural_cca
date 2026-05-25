Neural CCA
===================

.. image:: _static/logo_neural_cca_wide.svg
   :alt: Neural CCA
   :align: center
   :width: 600px

Spike sorting, spike train statistics, and orientation selectivity analysis
for extracellular neural recordings.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Spike Sorting
      :link: api/sorting
      :link-type: doc

      KMeans-based spike clustering with PCA, quality metrics
      (silhouette, SNR, ISI violations, isolation distance), and
      zarr export.

   .. grid-item-card:: Tuning Analysis
      :link: api/tuning
      :link-type: doc

      Orientation / direction selectivity indices, tuning-curve fitting
      (von Mises, double Gaussian), F0/F1/F2 harmonic modulation,
      population statistics, and bootstrap significance tests.

   .. grid-item-card:: Spike Train Analysis
      :link: api/sta
      :link-type: doc

      Mean firing rates, ISI statistics, CV, LvR, Fano factor,
      autocorrelograms, PSTHs, first-spike latency, and
      trial-to-trial reliability.

Quick start
-----------

.. code-block:: python

   from neural_cca import (
       load_from_arrays,
       run_sorting_pipeline,
       get_os_metrics,
       to_zarr_flat,
   )

   data   = load_from_arrays(waveforms, spike_times, trials, angles)
   result = run_sorting_pipeline(data)
   os     = get_os_metrics(
       spike_times,
       trials,
       angles,
       all_clusters=False,
       cluster_labels=result.cluster_labels,
       cluster_id=0,
   )

   # persist to zarr
   to_zarr_flat(result, data, "my_sorting.zarr")

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api/index
   developer
   changelog
