"""Deprecated shim — plotting lives in :mod:`neural_cca.spike_train.plotting`.

Importing this submodule emits a :class:`DeprecationWarning` because
the ``sta`` subpackage was renamed to ``spike_train`` in v0.2.0.
Every public symbol is re-exported here so
``from neural_cca.sta.plotting import X`` keeps working for now.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "`neural_cca.sta.plotting` was renamed to "
    "`neural_cca.spike_train.plotting` in v0.2.0; this shim will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from neural_cca.spike_train.plotting import *  # noqa: F401, F403, E402
