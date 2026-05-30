"""Per-trial post-stimulus spike filtering — built once, passed down.

The functions in this subpackage that compute orientation statistics
all need the same intermediate object: a per-trial dict of spikes
falling inside the stimulus window, with the corresponding mean firing
rate per trial and a copy of the angles array for alignment.

Building that object is O(n_spikes) and was previously inlined into
:func:`get_os_metrics`, :func:`anova_across_orientations`, and
:func:`modulation_ratio_per_orientation`, so a single
``get_os_metrics(compute_p_values=True, ...)`` call would walk the
spike arrays three or four times.

This module centralises the construction in :func:`_build_trial_filter`
and a small :class:`_TrialFilteredSpikes` dataclass.  The convention
inside the package is:

* :func:`get_os_metrics` calls ``_build_trial_filter`` exactly once at
  the top of the function.
* It then forwards the resulting object to every helper that would
  otherwise rebuild it (via the private ``_filter=`` keyword argument
  on each helper).
* Public consumers that are called directly (not via
  ``get_os_metrics``) still build their own filter internally and
  remain ergonomic for ad-hoc use.

Both the dataclass and the builder are package-private (leading
underscore) — they are an implementation detail, not part of the
public API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["_TrialFilteredSpikes", "_build_trial_filter"]


@dataclass
class _TrialFilteredSpikes:
    """Per-trial post-stimulus spikes for one cluster (or 'all spikes').

    Built once at the top of a metric pipeline and forwarded to every
    consumer to avoid rebuilding the same per-trial filter several
    times.  All times are in seconds; angles in degrees.

    Attributes:
        spike_times_by_trial: ``{trial_idx: spike_times_in_window}``.
            Each value is a 1-D ``float64`` array of trial-relative
            spike times that fall inside ``(stim_onset, stim_offset]``.
            Empty arrays are present for trials with no spikes so the
            mapping covers ``range(n_trials)`` exactly.
        mfrs: ``(n_trials,)`` mean firing rate per trial in Hz,
            computed as ``len(spikes) / stim_duration``.
        angles: ``(n_trials,)`` stimulus angle in degrees per trial,
            aligned with *mfrs*.
        trial_indices: ``(n_trials,)`` ``arange(n_trials)``, kept as a
            named field so a stratified bootstrap can resample by
            trial without recomputing.
        stim_onset: Stimulus onset in seconds (inclusive lower bound
            of the half-open spike window).
        stim_offset: Stimulus offset in seconds (inclusive upper
            bound — spikes at exactly *stim_offset* are kept).

    Notes:
        ``stim_duration`` is exposed as a property so callers do not
        store the same number twice.
    """

    spike_times_by_trial: dict[int, npt.NDArray[np.float64]]
    mfrs: npt.NDArray[np.float64]
    angles: npt.NDArray[np.float64]
    trial_indices: npt.NDArray[np.int64]
    stim_onset: float
    stim_offset: float

    @property
    def stim_duration(self) -> float:
        """Length of the stimulus window in seconds."""
        return self.stim_offset - self.stim_onset

    @property
    def n_trials(self) -> int:
        """Number of trials covered by *angles* / *mfrs*."""
        return int(self.angles.shape[0])


def _build_trial_filter(
    spike_times: npt.ArrayLike,
    trials: npt.ArrayLike,
    angles: npt.ArrayLike,
    # NOTE: Natal-specific default; v0.2.0 will make this required.
    stim_window: tuple[float, float] = (0.5, 2.5),
    cluster_labels: npt.ArrayLike | None = None,
    cluster_id: int | None = None,
) -> _TrialFilteredSpikes:
    """Build a :class:`_TrialFilteredSpikes` from raw spike arrays.

    Performs every cluster-and-window filtering step exactly once,
    materialising the per-trial dict, the per-trial firing rates, and
    the timing constants in a single :class:`_TrialFilteredSpikes`
    object that downstream metric functions can reuse.

    Args:
        spike_times: ``(n_spikes,)`` trial-relative spike times in
            seconds.
        trials: ``(n_spikes,)`` trial index per spike.
        angles: ``(n_trials,)`` stimulus angle in degrees per trial.
            ``len(angles)`` defines the trial count.
        stim_window: ``(onset, offset)`` half-open stimulus interval
            in seconds.  Spikes with ``onset < t <= offset`` are kept.
        cluster_labels: Optional ``(n_spikes,)`` cluster label per
            spike.  Must be provided together with *cluster_id*.
        cluster_id: Optional cluster ID to restrict the filter to.
            Both *cluster_labels* and *cluster_id* must be ``None``
            (use all spikes) or both non-``None`` (filter by cluster).

    Returns:
        :class:`_TrialFilteredSpikes` with the per-trial dict, MFRs,
        angles, trial indices, and timing constants.

    Raises:
        ValueError: If *cluster_labels* and *cluster_id* are not both
            provided together.
    """
    if (cluster_labels is None) != (cluster_id is None):
        raise ValueError(
            "cluster_labels and cluster_id must both be provided together, or both omitted."
        )

    spike_times = np.asarray(spike_times, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.int64)
    angles = np.asarray(angles, dtype=np.float64)
    s_on, s_off = stim_window

    # Enforce the trial-index contract.  ``angles[k]`` is taken to be
    # the stimulus angle of trial ``k``, so every value in ``trials``
    # must be a valid index into ``angles``.  Without this check, a
    # user that hands in ``trials = [10, 11, 12]`` with a 3-element
    # ``angles`` array gets silently wrong per-trial firing rates
    # because the builder iterates ``range(len(angles))`` and never
    # sees the supplied trial IDs.
    n_trials = int(angles.shape[0])
    if trials.size > 0:
        t_min = int(trials.min())
        t_max = int(trials.max())
        if t_min < 0 or t_max >= n_trials:
            raise ValueError(
                "Trial indices must lie in [0, len(angles)); got "
                f"min={t_min}, max={t_max}, len(angles)={n_trials}. "
                "The package convention is that `angles[k]` is the "
                "stimulus angle of trial `k`, so non-contiguous or "
                "out-of-range trial IDs are not supported."
            )

    # Apply cluster filter once.  After this point, every spike in
    # spike_times / trials belongs to the requested cluster (or all
    # clusters if no filter was given).
    if cluster_labels is not None and cluster_id is not None:
        cluster_labels = np.asarray(cluster_labels)
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        trials = trials[mask]

    # Apply stimulus-window filter once, then bucket by trial in a
    # single pass.  This is the only place in the package that walks
    # the raw spike arrays for orientation analyses.
    in_window = (spike_times > s_on) & (spike_times <= s_off)
    windowed_st = spike_times[in_window]
    windowed_tr = trials[in_window]

    spike_times_by_trial: dict[int, npt.NDArray[np.float64]] = {
        t: windowed_st[windowed_tr == t] for t in range(n_trials)
    }

    duration = float(s_off - s_on)
    if duration > 0:
        mfrs = np.array(
            [len(spike_times_by_trial[t]) / duration for t in range(n_trials)],
            dtype=np.float64,
        )
    else:
        mfrs = np.zeros(n_trials, dtype=np.float64)

    return _TrialFilteredSpikes(
        spike_times_by_trial=spike_times_by_trial,
        mfrs=mfrs,
        angles=angles,
        trial_indices=np.arange(n_trials, dtype=np.int64),
        stim_onset=float(s_on),
        stim_offset=float(s_off),
    )
