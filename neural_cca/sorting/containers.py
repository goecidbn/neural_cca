"""Data containers for the spike-sorting pipeline.

Holds :class:`SortingData` (input) and :class:`SortingResult` (output)
dataclasses.  These are pure value objects with no I/O concerns; they
live in their own module so :mod:`sorting.sorting`,
:mod:`sorting.io_util`, :mod:`sorting.batch`, and
:mod:`sorting.plotting` can all import them without forming a
cycle.

Earlier revisions kept the dataclasses inside ``io_util.py`` to avoid a
circular import with ``sorting.py``; that constraint is gone now and
the package layout reflects what each module is actually responsible
for.

Public re-exports are provided from :mod:`sorting` so existing
``from neural_cca import SortingData`` imports still work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = [
    "SortingData",
    "SortingResult",
]


@dataclass
class SortingResult:
    """Output of the spike-sorting pipeline.

    Attributes:
        cluster_labels: ``(n_spikes,)`` cluster assignment per spike.
        n_clusters: Number of clusters.
        quality: Dict of sorting-quality metrics (SNR, RPVs, etc.).
        os_metrics: Per-cluster orientation-selectivity metrics
            (``{cluster_id: {...}}``).  ``None`` when angles are absent
            or tuning is not available.
        k_search: Dict with silhouette scores per *k* tested
            (``None`` when *k* was provided directly).
        metadata: Extra information (KMeans params, etc.).
    """

    cluster_labels: npt.NDArray[np.int64]
    n_clusters: int
    quality: dict
    os_metrics: dict[int, dict] | None = None
    k_search: dict | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SortingData:
    """Minimal container for spike-sorting input data.

    All arrays use the convention that each *row* is one spike event.

    Attributes:
        waveforms: (n_spikes, snippet_length) waveform snippets.
        spike_times: (n_spikes,) spike time within each trial (seconds).
        trials: (n_spikes,) trial index (0-based) for each spike.
        angles: (n_trials,) stimulus angle in degrees for each trial.
        waveform_fs: Waveform sampling rate in Hz (e.g. 32 000).
        n_trials: Total number of trials (may exceed
            ``len(unique(trials))`` if some trials had no spikes).
        stim_window: ``(onset, end)`` of the stimulus period within
            each trial (seconds).  Must satisfy ``onset < end`` —
            ``__post_init__`` raises ``ValueError`` on a zero or
            inverted window so a typo doesn't silently divide by
            zero in the downstream firing-rate calculation.  Spikes
            that fall in ``(onset, end]`` are part of the stimulated
            portion; ``end`` is also the assumed full trial length.
        stim_frequency: Temporal frequency of the visual stimulus
            (Hz).  Set to ``None`` when unknown / not applicable.
        metadata: Arbitrary extra information (electrode id, animal, …).
    """

    waveforms: npt.NDArray[np.float64]
    spike_times: npt.NDArray[np.float64]
    trials: npt.NDArray[np.int64]
    angles: npt.NDArray[np.float64]
    waveform_fs: float = 32_000.0
    n_trials: int | None = None
    stim_window: tuple[float, float] = (0.5, 2.5)
    stim_frequency: float | None = 2.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.waveforms)
        if len(self.spike_times) != n:
            raise ValueError(
                f"waveforms ({n}) and spike_times ({len(self.spike_times)}) "
                "must have the same length."
            )
        if len(self.trials) != n:
            raise ValueError(
                f"waveforms ({n}) and trials ({len(self.trials)}) must have the same length."
            )
        if self.n_trials is None:
            self.n_trials = int(len(self.angles))
        # Normalise to a tuple of two floats — accept lists / arrays as
        # well so deserialised JSON / Zarr attrs round-trip cleanly.
        s_on, s_end = self.stim_window
        s_on, s_end = float(s_on), float(s_end)
        # A non-positive stimulus duration would silently divide by zero
        # downstream (firing-rate calc in ``batch.py``) and yield NaN
        # rates with no breadcrumb — catch the typo at construction time.
        if s_end <= s_on:
            raise ValueError(
                f"stim_window must satisfy onset < end (got {(s_on, s_end)}); "
                "a zero or negative duration produces NaN firing rates."
            )
        self.stim_window = (s_on, s_end)

    @property
    def n_spikes(self) -> int:
        """Total number of spike events."""
        return len(self.waveforms)

    @property
    def snippet_length(self) -> int:
        """Number of samples per waveform snippet."""
        return self.waveforms.shape[1]

    @property
    def time_axis_ms(self) -> npt.NDArray[np.float64]:
        """Time axis for one waveform snippet in milliseconds."""
        return np.arange(self.snippet_length) / self.waveform_fs * 1000.0

    @property
    def stimulus_duration(self) -> float:
        """Duration of the stimulus period (``stim_window[1] - stim_window[0]``)."""
        return self.stim_window[1] - self.stim_window[0]
