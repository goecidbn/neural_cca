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

   .. grid-item-card:: Spike Train Statistics
      :link: api/spike_train
      :link-type: doc

      Mean firing rates, ISI statistics, CV, LvR, Fano factor,
      autocorrelograms, PSTHs, first-spike latency (incl. the
      response-detection ``_thresholded`` variant), and
      trial-to-trial reliability.  Previously called ``sta``.

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


----------------

Institutions & Funding
======================

.. raw:: html

   <div class="logo-grid">
     <a href="https://uni-goettingen.de/en/608362.html" target="_blank">
       <img src="_static/logo_cidbn.jpg" alt="CIDBN / University of Göttingen" />
     </a>
     <a href="https://www.mwk.niedersachsen.de/startseite/" target="_blank">
       <img src="_static/logo_mwk.png" alt="Niedersächsisches Ministerium für Wissenschaft und Kultur" />
     </a>
     <a href="https://www.daad.de/en/" target="_blank">
       <img src="_static/logo_daad.png" alt="DAAD — German Academic Exchange Service" />
     </a>
   </div>

The developer team is part of the `Göttingen Campus Institute for Dynamics of Biological Networks (CIDBN) <https://uni-goettingen.de/en/608362.html>`_.
This project is partially supported by the
`Niedersächsisches Ministerium für Wissenschaft und Kultur (MWK) <https://www.mwk.niedersachsen.de/startseite/>`_
and the `DAAD (Deutscher Akademischer Austauschdienst) <https://www.daad.de/en/>`_
through a PPP Brazil PROBRAL collaboration with `CAPES <https://www.gov.br/capes/>`_ (Project ID: 57754779).
