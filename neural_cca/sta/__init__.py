"""Deprecated alias for :mod:`neural_cca.spike_train`.

The subpackage was renamed to ``spike_train`` in v0.2.0 to free the
``sta`` name for the *spike-triggered average* (Schwartz et al. 2006)
analysis that "STA" canonically denotes in the computational-
neuroscience literature.  This shim re-exports every name from
``spike_train`` and emits a :class:`DeprecationWarning` on first
import so existing scripts keep working unchanged.

Migrate by replacing ``neural_cca.sta`` → ``neural_cca.spike_train``
in every import path.  The function APIs are unchanged.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "`neural_cca.sta` was renamed to `neural_cca.spike_train` in "
    "v0.2.0.  The old import path is a deprecated shim and will be "
    "removed in a future release.  Update your imports: "
    "`from neural_cca.spike_train import ...`.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the full public surface so ``from neural_cca.sta import X``
# keeps working byte-for-byte during the deprecation window.
from neural_cca.spike_train import *  # noqa: F401, F403, E402
from neural_cca.spike_train import __all__ as _spike_train_all  # noqa: E402
from neural_cca.spike_train import analysis, plotting  # noqa: F401, E402

__all__ = list(_spike_train_all)
