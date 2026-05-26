:orphan:

sta — Deprecated alias for ``spike_train``
==========================================

.. deprecated:: 0.1.3

   The ``sta`` subpackage was renamed to
   :doc:`spike_train <spike_train>`.  The old import path
   (``neural_cca.sta.*``) still works as a thin shim that emits a
   :class:`DeprecationWarning` on import.  The rename frees the
   ``sta`` name for the canonical *spike-triggered average*
   (Schwartz et al. 2006) meaning.  Migrate by replacing
   ``neural_cca.sta`` → ``neural_cca.spike_train`` in every import.
