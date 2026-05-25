"""Deprecated shim — analysis lives in :mod:`neural_cca.spike_train.analysis`.

Importing this submodule emits a :class:`DeprecationWarning` because
the ``sta`` subpackage was renamed to ``spike_train`` in v0.2.0
(``STA`` canonically denotes spike-triggered average, not
spike-train statistics).  Every public symbol is re-exported here so
``from neural_cca.sta.analysis import X`` keeps working for now.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "`neural_cca.sta.analysis` was renamed to "
    "`neural_cca.spike_train.analysis` in v0.2.0; this shim will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from neural_cca.spike_train.analysis import *  # noqa: F401, F403, E402
from neural_cca.spike_train.analysis import __all__ as _all  # noqa: E402

__all__ = list(_all)
