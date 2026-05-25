"""Spike train analysis — MFR, CV, LvR, autocorrelogram, PSTH, and more.

Adapted from the patch-clamp reference implementation
(``sta_other.py``) but operates on **pre-sorted spike times**
from extracellular recordings rather than raw voltage traces.

Reference for CV and LvR:
    https://gist.github.com/fschwar4/8e9044273716cfea5a76653daeb0d170
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats

# Cross-package import lifted to module top.  ``tuning.temporal``
# is a sibling subpackage with no back-edge to ``sta``, so this does
# not introduce a circular import.  See ``trial_to_trial_reliability``
# (stat="f1_phase") for the only consumer.
from ..tuning.temporal import f1_phase as _f1_phase

__all__ = [
    "minimal_spike_train_analysis",
    "calc_mfr_trial",
    "isi_violation_rate",
    "firing_rate_stability",
    "autocorrelogram",
    "fano_factor",
    "local_variation",
    "cv_log_isi",
    "psth",
    "trial_to_trial_reliability",
    "trial_to_trial_correlation_matrix",
    "first_spike_latency",
]


# ---------------------------------------------------------------------------
# Helper: filter spike times by cluster
# ---------------------------------------------------------------------------


def _filter_cluster(
    spike_times: npt.NDArray,
    cluster_labels: npt.NDArray | None,
    cluster_id: int | None,
) -> npt.NDArray:
    """Return spike times for a single cluster (or all if not filtered)."""
    if cluster_labels is not None and cluster_id is not None:
        return spike_times[cluster_labels == cluster_id]
    return spike_times


def _positive_isis(spike_times: npt.NDArray) -> npt.NDArray:
    """ISIs excluding negative values (trial-boundary artefacts).

    .. warning::
       Only correct on a spike train concatenated *trial-by-trial in
       trial order* (so the diffs at trial boundaries are negative and
       get filtered out).  Will silently include cross-trial pairs as
       fake ISIs if the input was globally sorted by trial-relative
       time (which is what ``np.argsort(spike_times)`` produces on
       trial-relative data).  Prefer :func:`_per_trial_isis` whenever
       a ``trials`` array is available.
    """
    isis = np.diff(spike_times)
    return isis[isis > 0]


# ---------------------------------------------------------------------------
# Trial-aware ISI helpers
# ---------------------------------------------------------------------------
#
# These helpers are the single source of truth for "what counts as an ISI"
# on trial-based data.  Every public function that operates on inter-spike
# intervals (CV, LV, LvR, CV-log-ISI, ISI violation rate, …) routes
# through them so cross-trial spike pairs are never silently treated as
# real ISIs.
#
# Trial-relative spike times restart at zero every trial, and the
# example notebooks sort the data globally by spike_times.  Both
# ``np.diff(spike_times)`` and the legacy ``_positive_isis`` then return
# values that mix within-trial and across-trial gaps; the resulting
# pseudo-ISIs are typically dominated by cross-trial pairs (~50 % of
# them sit below 1 ms for a moderately dense recording).  See the bug
# tracker entry for "ISI rate of 600 Hz on a 4 Hz cell" for the full
# failure mode.


def _per_trial_isis(
    spike_times: npt.NDArray,
    trials: npt.NDArray | None,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
) -> list[np.ndarray]:
    """Within-trial ISIs as a list of per-trial arrays.

    Each element is the array of *positive* ISIs computed *within* one
    trial (after sorting that trial's spikes chronologically), so
    cross-trial pairs are structurally excluded.  Trials with fewer
    than two spikes contribute no entry.

    For continuous (non-trial) recordings pass ``trials=None``; the
    function then sorts globally and returns a single ISI array as a
    one-element list, so callers can iterate uniformly.
    """
    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        if trials is not None:
            trials = trials[mask]

    if trials is None:
        st = np.sort(np.asarray(spike_times, dtype=np.float64))
        if len(st) < 2:
            return []
        d = np.diff(st)
        d = d[d > 0]
        return [d] if len(d) > 0 else []

    spike_times = np.asarray(spike_times, dtype=np.float64)
    trials = np.asarray(trials)
    out: list[np.ndarray] = []
    for t in np.unique(trials):
        spk = np.sort(spike_times[trials == t])
        if len(spk) < 2:
            continue
        d = np.diff(spk)
        if len(d) > 0:
            out.append(d)
    return out


def _pooled_cv(per_trial_isis: list[np.ndarray]) -> float:
    """CV computed from all within-trial ISIs pooled into one sample."""
    if not per_trial_isis:
        return float("nan")
    all_isis = np.concatenate(per_trial_isis)
    if len(all_isis) < 2:
        return float("nan")
    m = float(np.mean(all_isis))
    if m == 0:
        return float("nan")
    return float(np.std(all_isis) / m)


def _pooled_cv_log_isi(per_trial_isis: list[np.ndarray]) -> float:
    """CV of log10(ISI) pooled across all within-trial ISIs."""
    if not per_trial_isis:
        return float("nan")
    all_isis = np.concatenate(per_trial_isis)
    if len(all_isis) < 2:
        return float("nan")
    log_isis = np.log10(all_isis)
    mean_log = float(np.mean(log_isis))
    if abs(mean_log) < 1e-12:
        return float("nan")
    return float(np.std(log_isis) / abs(mean_log))


def _pooled_lv(per_trial_isis: list[np.ndarray]) -> float:
    """Local variation pooled across trials with correct pair weighting.

    LV is defined as ``(3/(n-1)) * Σ ((ISI_i − ISI_{i+1})/(ISI_i + ISI_{i+1}))²``
    over the *consecutive pairs* of one ISI sequence.  Pooling across
    trials means summing the per-pair contributions over every trial's
    consecutive pairs and dividing by the *total* number of pairs.
    Cross-trial pairs are never created, so the spurious "boundary
    pairs" you would get from a flat concatenation never enter the
    sum.
    """
    total_num = 0.0
    total_pairs = 0
    for isis in per_trial_isis:
        if len(isis) < 2:
            continue
        num = (isis[:-1] - isis[1:]) ** 2
        denom = (isis[:-1] + isis[1:]) ** 2
        valid = denom > 0
        if not np.any(valid):
            continue
        total_num += float(np.sum(num[valid] / denom[valid]))
        total_pairs += int(np.sum(valid))
    if total_pairs == 0:
        return float("nan")
    return 3.0 * total_num / total_pairs


def _pooled_lvr(
    per_trial_isis: list[np.ndarray],
    refractory_period: float = 0.001,
) -> float:
    """LvR pooled across trials, refractory-corrected.

    Same accounting as :func:`_pooled_lv`: every per-trial pair
    contributes one term, and we divide by the total number of pairs
    instead of any individual trial's count.
    """
    total_num = 0.0
    total_pairs = 0
    for isis in per_trial_isis:
        if len(isis) < 2:
            continue
        si_ = isis[:-1] + isis[1:]
        valid = si_ > 0
        if not np.any(valid):
            continue
        ft = 1.0 - (4.0 * isis[:-1] * isis[1:]) / si_**2
        st_term = 1.0 + (4.0 * refractory_period) / si_
        contrib = ft * st_term
        total_num += float(np.sum(contrib[valid]))
        total_pairs += int(np.sum(valid))
    if total_pairs == 0:
        return float("nan")
    return 3.0 * total_num / total_pairs


# ---------------------------------------------------------------------------
# Per-trial firing rate (moved from tuning.trial_rates)
# ---------------------------------------------------------------------------


def calc_mfr_trial(
    spike_times: npt.NDArray[np.float64],
    trials: npt.NDArray[np.int64],
    all_clusters: bool = True,
    cluster_labels: npt.NDArray[np.int64] | None = None,
    cluster_id: int | None = None,
    stim_window: tuple[float, float] = (0.5, 2.5),
    n_trials: int | None = None,
) -> dict[int, float]:
    """Mean firing rate per trial (during stimulus period).

    Args:
        spike_times: Spike times (trial-relative, seconds).
        trials: Trial index per spike.
        all_clusters: If ``True`` use all spikes; if ``False`` filter
            by *cluster_id*.
        cluster_labels: Cluster labels per spike (required when
            ``all_clusters=False``).
        cluster_id: Cluster to select (required when
            ``all_clusters=False``).
        stim_window: ``(onset, end)`` of the stimulus period within
            each trial (seconds). The mean firing rate is computed
            from spikes that fall in ``(onset, end]`` and divided by
            ``end - onset``.
        n_trials: Number of trials.  Defaults to
            ``len(unique(trials))``.

    Returns:
        Dict mapping trial index to mean firing rate (Hz).

    Raises:
        ValueError: If ``cluster_labels`` and ``cluster_id`` are not
            both provided when ``all_clusters=False``.
    """
    if not all_clusters and (cluster_labels is None or cluster_id is None):
        raise ValueError("cluster_labels and cluster_id required when all_clusters is False.")

    s_on, s_end = stim_window
    stim_duration = s_end - s_on
    n_trials = n_trials if n_trials is not None else len(np.unique(trials))

    mfr_by_trial: dict[int, float] = {}
    for trial_idx in np.arange(n_trials):
        trial_mask = trials == trial_idx
        sts = spike_times[trial_mask]

        if all_clusters:
            n_spikes = np.sum((sts > s_on) & (sts <= s_end))
        else:
            n_spikes = np.sum(
                (sts > s_on) & (sts <= s_end) & (cluster_labels[trial_mask] == cluster_id)
            )

        mfr_by_trial[trial_idx] = float(n_spikes) / stim_duration

    return mfr_by_trial


# ---------------------------------------------------------------------------
# Original function (unchanged)
# ---------------------------------------------------------------------------


def minimal_spike_train_analysis(
    spike_times: npt.NDArray[np.float64],
    trials: npt.NDArray[np.int64] | None = None,
    cluster_labels: npt.NDArray[np.int64] | None = None,
    cluster_id: int | None = None,
    refractory_period: float = 0.001,
    stim_window: tuple[float, float] = (0.5, 2.5),
    n_trials: int = 240,
    only_spontaneous: bool = False,
    only_stimulated: bool = False,
) -> dict[str, float]:
    """Compute basic spike-train statistics for one cluster.

    Unlike the patch-clamp version that detects peaks in voltage traces,
    this function operates on already-detected spike times from
    extracellular spike sorting.

    For trial-based recordings, pass the *trials* array.  CV and LvR
    are then computed from *within-trial* ISIs only — never from the
    cross-trial gaps that a global ``np.diff`` would otherwise pick up.

    Args:
        spike_times: Spike times in seconds (trial-relative).
        trials: Optional trial index per spike.  When given, ISIs are
            computed within each trial only and the result mirrors the
            data layout the rest of ``sta`` expects.
        cluster_labels: Cluster assignment per spike.  If provided
            together with *cluster_id*, only spikes from that cluster
            are analysed.
        cluster_id: Which cluster to analyse.  Both *cluster_labels*
            and *cluster_id* must be provided together, or both omitted.
        refractory_period: Refractory period for LvR computation
            (seconds).
        stim_window: ``(onset, end)`` of the stimulus period within
            a trial (seconds). The trial is assumed to span
            ``[0, end]``: ``onset`` separates the spontaneous and
            stimulated portions, and ``end`` is the full trial length
            used for total-recording-duration calculations.
        n_trials: Number of experimental trials (used to convert spike
            count to MFR; ignored when *trials* is given because the
            unique trial count is then derived from the data).
        only_spontaneous: Analyse only pre-stimulus spikes
            (``[0, onset)``).
        only_stimulated: Analyse only post-onset spikes
            (``[onset, end]``).

    Returns:
        Dictionary with keys ``"mfr"`` (Hz), ``"cv"`` (dimensionless),
        and ``"lvr"`` (dimensionless).  Values are ``np.nan`` when
        there are insufficient spikes for the computation.

    Raises:
        ValueError: If *only_spontaneous* and *only_stimulated* are
            both ``True``, or if only one of *cluster_labels* /
            *cluster_id* is provided.
    """
    if only_spontaneous and only_stimulated:
        raise ValueError("Cannot analyse both only_spontaneous and only_stimulated simultaneously.")
    if (cluster_labels is None) != (cluster_id is None):
        raise ValueError(
            "Both cluster_labels and cluster_id must be provided together, or both omitted."
        )

    s_on, s_end = stim_window

    # --- Filter by cluster ---
    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        if trials is not None:
            trials = trials[mask]

    # --- Filter by stimulus period ---
    if only_spontaneous:
        win_mask = spike_times < s_on
        per_trial_win = s_on
    elif only_stimulated:
        win_mask = spike_times >= s_on
        per_trial_win = s_end - s_on
    else:
        win_mask = np.ones(len(spike_times), dtype=bool)
        per_trial_win = s_end

    spike_times = spike_times[win_mask]
    if trials is not None:
        trials = trials[win_mask]

    # --- Mean firing rate ---
    n_trials_eff = int(len(np.unique(trials))) if trials is not None else int(n_trials)
    duration = n_trials_eff * per_trial_win
    mfr = len(spike_times) / duration if duration > 0 else 0.0

    # --- Within-trial ISIs (cross-trial pairs are structurally excluded) ---
    per_trial_isis = _per_trial_isis(spike_times, trials)
    if not per_trial_isis or sum(len(x) for x in per_trial_isis) < 2:
        return {"mfr": mfr, "cv": float("nan"), "lvr": float("nan")}

    cv = _pooled_cv(per_trial_isis)
    lvr = _pooled_lvr(per_trial_isis, refractory_period=refractory_period)

    return {"mfr": mfr, "cv": cv, "lvr": lvr}


# ---------------------------------------------------------------------------
# ISI violation rate
# ---------------------------------------------------------------------------


def isi_violation_rate(
    spike_times: npt.NDArray,
    trials: npt.NDArray | None = None,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    refractory_period: float = 0.001,  # seconds
    trial_duration: float | None = None,
    return_percentage: bool = False,
) -> float:
    """ISI violation rate.

    Counts inter-spike intervals shorter than *refractory_period* and
    expresses them as either a rate (Hz) or a percentage of ISIs.

    For **trial-based recordings** (the common case in this package),
    pass the *trials* array.  The function then:

    1. Computes ISIs *within each trial only*, so that two spikes from
       different trials can never form a "violation" — their gap is
       the difference of two trial-relative timestamps and has no
       physical meaning as an ISI.
    2. Uses the correct total recording duration of
       ``n_trials * trial_duration`` instead of
       ``max(spike_times) - min(spike_times)``, which is at most the
       size of one trial window and inflates the rate by ~``n_trials``.

    For **continuous (non-trial) recordings**, omit *trials*. The
    function then sorts the spikes globally, takes positive ISIs, and
    uses ``max - min`` of the spike train as the duration.

    Args:
        spike_times: Spike times (seconds). Trial-relative if *trials*
            is given; otherwise absolute.
        trials: Optional trial index per spike. When provided, ISIs are
            computed within each trial only and the recording duration
            is taken as ``n_unique_trials * trial_duration``.
        cluster_labels: Optional cluster labels for filtering.
        cluster_id: Cluster to analyse.
        refractory_period: Refractory period (seconds, default 1 ms).
        trial_duration: Length of one trial in seconds.  Required when
            *trials* is given (and ignored otherwise).
        return_percentage: If ``True``, return the percentage of ISIs
            that are violations instead of a rate in Hz.

    Returns:
        Violation rate (violations per second) by default, or
        percentage of ISIs that are violations when
        ``return_percentage=True``.  Returns ``0.0`` when there are
        fewer than 2 valid spikes.

    Raises:
        ValueError: If *trials* is given without *trial_duration*.
    """
    # Filter by cluster
    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        if trials is not None:
            trials = trials[mask]

    if len(spike_times) < 2:
        return 0.0

    if trials is not None:
        if trial_duration is None:
            raise ValueError("trial_duration is required when 'trials' is provided.")
        n_violations = 0
        n_isis = 0
        for t in np.unique(trials):
            spk = np.sort(spike_times[trials == t])
            if len(spk) < 2:
                continue
            diffs = np.diff(spk)
            n_violations += int(np.sum(diffs < refractory_period))
            n_isis += len(diffs)
        n_unique_trials = int(len(np.unique(trials)))
        duration = n_unique_trials * float(trial_duration)
    else:
        # Continuous recording — sort globally and use the spike-train
        # span as the duration.  Negative diffs (which can appear if
        # the user accidentally passes trial-relative spike times
        # without ``trials``) are excluded from the violation count.
        st = np.sort(spike_times)
        diffs = np.diff(st)
        diffs = diffs[diffs > 0]
        n_violations = int(np.sum(diffs < refractory_period))
        n_isis = int(len(diffs))
        duration = float(st.max() - st.min())

    if return_percentage:
        if n_isis == 0:
            return 0.0
        return (n_violations / n_isis) * 100.0

    if duration <= 0:
        return 0.0
    return n_violations / duration


# ---------------------------------------------------------------------------
# Firing rate stability
# ---------------------------------------------------------------------------


def firing_rate_stability(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    window_size: float = 0.5,
    stat: str = "mean",
    trial_duration: float = 2.5,
    refractory_period: float = 0.001,
) -> dict[str, float | npt.NDArray]:
    """Firing-rate statistic across sliding time windows.

    Divides the trial time axis into non-overlapping windows of
    *window_size* seconds and computes the requested *stat* in each
    window.

    For ISI-derived statistics (``"cv"``, ``"logcv"``, ``"lv"``,
    ``"lvr"``), inter-spike intervals are computed *within each trial*
    and then pooled across trials for each window — cross-trial spike
    pairs are never counted.  The ``trials`` array is therefore
    required whenever you use one of these stats; without it, the
    function would fall back to ``np.diff`` on the globally-pooled
    spike train, which produces contaminated pseudo-ISIs on
    trial-relative data (see the trial-relative bug class
    documented in :func:`autocorrelogram` and
    :func:`isi_violation_rate`).

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.  Used by all stats: the
            ``"mean"`` path divides by ``n_trials * window_size`` to
            get a rate, and the ISI-derived stats (``"cv"``,
            ``"logcv"``, ``"lv"``, ``"lvr"``) need it to compute
            within-trial ISIs.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to analyse.
        window_size: Window duration (seconds).
        stat: Statistic to compute per window.  One of ``"mean"``
            (firing rate), ``"cv"``, ``"logcv"``, ``"lv"``, ``"lvr"``,
            ``"fano"``.
        trial_duration: Duration of one trial (seconds).
        refractory_period: Refractory period in seconds, forwarded to
            ``_pooled_lvr`` when ``stat="lvr"``.  Ignored otherwise.

    Returns:
        Dict with ``"values"`` (array, one per window), ``"mean"``,
        ``"std"``, and ``"cv_of_stat"`` (stability measure).
    """
    if cluster_labels is not None and cluster_id is not None:
        cl_mask = cluster_labels == cluster_id
        spike_times = spike_times[cl_mask]
        if trials is not None:
            trials = trials[cl_mask]
    st = np.asarray(spike_times)
    if trials is not None:
        trials = np.asarray(trials)

    n_windows = max(1, int(trial_duration / window_size))
    edges = np.linspace(0, trial_duration, n_windows + 1)

    def _win_per_trial_isis(lo: float, hi: float) -> list[np.ndarray]:
        """Within-trial ISIs whose *both* spikes fall in ``[lo, hi)``.

        For ISI-based statistics like CV / LV / LvR we need pairs of
        consecutive spikes from the *same* trial that both lie in the
        window — never cross-trial pairs and never pairs that straddle
        the window boundary.
        """
        if trials is None:
            spk = np.sort(st[(st >= lo) & (st < hi)])
            if len(spk) < 2:
                return []
            d = np.diff(spk)
            d = d[d > 0]
            return [d] if len(d) > 0 else []
        out: list[np.ndarray] = []
        for t in np.unique(trials):
            spk = np.sort(st[(trials == t) & (st >= lo) & (st < hi)])
            if len(spk) < 2:
                continue
            d = np.diff(spk)
            if len(d) > 0:
                out.append(d)
        return out

    values: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if stat == "mean":
            n_trials_total = int(len(np.unique(trials))) if trials is not None else 1
            dur = (hi - lo) * n_trials_total
            n_in_win = int(np.sum((st >= lo) & (st < hi)))
            values.append(n_in_win / dur if dur > 0 else 0.0)
        elif stat == "cv":
            values.append(_pooled_cv(_win_per_trial_isis(lo, hi)))
        elif stat == "logcv":
            values.append(_pooled_cv_log_isi(_win_per_trial_isis(lo, hi)))
        elif stat == "lv":
            values.append(_pooled_lv(_win_per_trial_isis(lo, hi)))
        elif stat == "lvr":
            values.append(
                _pooled_lvr(
                    _win_per_trial_isis(lo, hi),
                    refractory_period=refractory_period,
                )
            )
        elif stat == "fano":
            if trials is not None:
                mask = (st >= lo) & (st < hi)
                counts = np.array(
                    [int(np.sum(mask & (trials == t))) for t in np.unique(trials)], dtype=np.float64
                )
                m = counts.mean()
                values.append(float(counts.var() / m) if m > 0 else float("nan"))
            else:
                values.append(float("nan"))
        else:
            raise ValueError(f"Unknown stat: {stat!r}")

    arr = np.array(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    mean_val = float(np.mean(valid)) if len(valid) > 0 else np.nan
    std_val = float(np.std(valid)) if len(valid) > 0 else np.nan
    cv_of = std_val / abs(mean_val) if mean_val != 0 and not np.isnan(mean_val) else np.nan

    return {"values": arr, "mean": mean_val, "std": std_val, "cv_of_stat": cv_of}


def _compute_lv(isis: npt.NDArray) -> float:
    """Local variation (LV) from ISIs."""
    if len(isis) < 2:
        return np.nan
    n = len(isis)
    num = (isis[:-1] - isis[1:]) ** 2
    denom = (isis[:-1] + isis[1:]) ** 2
    # Avoid division by zero
    valid = denom > 0
    if valid.sum() == 0:
        return np.nan
    return float((3.0 / (n - 1)) * np.sum(num[valid] / denom[valid]))


def _compute_lvr(isis: npt.NDArray, refractory_period: float = 0.001) -> float:
    """Local variation ratio (LvR) with refractory correction."""
    if len(isis) < 2:
        return np.nan
    n = len(isis)
    s_ = 3.0 / (n - 1)
    si_ = isis[:-1] + isis[1:]
    ft_ = 1.0 - (4.0 * isis[:-1] * isis[1:]) / si_**2
    st_ = 1.0 + (4.0 * refractory_period) / si_
    return float(s_ * np.sum(ft_ * st_))


# ---------------------------------------------------------------------------
# Autocorrelogram
# ---------------------------------------------------------------------------


def autocorrelogram(
    spike_times: npt.NDArray,
    trials: npt.NDArray | None = None,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    bin_size: float = 0.001,
    max_lag: float = 0.05,
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""Autocorrelogram of a spike train.

    Histogram of all pairwise spike-time differences within
    ±\ *max_lag*.

    For **trial-based recordings**, pass *trials* so that pairs are
    accumulated *within each trial only*.  Without it, this function
    sorts the spike train globally and counts pairs across all spikes
    within ``max_lag`` of each other — which on trial-relative data
    silently turns into an autocorrelogram of the *trial structure*
    (every cross-trial pair whose trial-relative timestamps happen to
    sit within ``max_lag`` is added), inflating the bin counts by
    roughly ``n_trials`` and washing out the real refractory dip and
    burstiness pattern.

    Args:
        spike_times: Spike times (seconds).  Trial-relative if *trials*
            is given; otherwise absolute.
        trials: Optional trial index per spike.  When given, pairs are
            accumulated within each trial only.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to analyse.
        bin_size: Bin width (seconds, default 1 ms).
        max_lag: Maximum lag (seconds, default 50 ms).

    Returns:
        ``(lags, counts)`` — bin centres and histogram counts. Self-pairs
        (``i == j``) are structurally excluded by the ``j = i + 1``
        bound, so the zero-lag bin only contains genuine coincidences
        between *distinct* spikes that happen to fall within one bin.
    """
    # Filter by cluster
    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        if trials is not None:
            trials = trials[mask]

    # Build edges from ``bin_size`` directly so each bin has width
    # exactly ``bin_size`` regardless of how ``max_lag / bin_size``
    # rounds.  ``n_half`` is the number of bins on each side of zero;
    # the effective range becomes ``[-n_half*bin_size, +n_half*bin_size]``
    # which differs from the requested ``max_lag`` by at most one bin
    # but preserves exact bin width and a perfectly symmetric layout
    # around lag zero.
    n_half = int(np.ceil(max_lag / bin_size))
    n_bins = 2 * n_half
    effective_max_lag = n_half * bin_size
    edges = np.linspace(-effective_max_lag, effective_max_lag, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)

    def _accumulate(st_sorted: np.ndarray) -> None:
        # Iterate over ordered pairs (i, j>i) of distinct spikes within
        # the effective range.  For each pair we add both +diff and
        # -diff so the symmetric histogram is filled in one pass.
        # i == j is impossible by construction, so no self-pairs ever
        # enter ``counts``.
        for i in range(len(st_sorted)):
            j = i + 1
            while j < len(st_sorted) and (st_sorted[j] - st_sorted[i]) <= effective_max_lag:
                diff = st_sorted[j] - st_sorted[i]
                counts[:] = (
                    counts
                    + np.histogram(
                        [diff, -diff],
                        bins=edges,
                    )[0]
                )
                j += 1

    if trials is None:
        # Continuous mode: sort globally and accumulate all pairs.
        _accumulate(np.sort(np.asarray(spike_times)))
    else:
        # Trial-aware: accumulate within each trial only.  Cross-trial
        # spike pairs have no physical meaning when ``spike_times`` is
        # trial-relative.
        trials = np.asarray(trials)
        spike_times = np.asarray(spike_times)
        for t in np.unique(trials):
            _accumulate(np.sort(spike_times[trials == t]))

    lags = 0.5 * (edges[:-1] + edges[1:])
    return lags, counts


# ---------------------------------------------------------------------------
# Fano factor
# ---------------------------------------------------------------------------

FanoMode = Literal["per_trial", "per_bin"]


def fano_factor(
    spike_times: npt.NDArray,
    trials: npt.NDArray | None = None,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    bin_size: float = 0.01,
    trial_duration: float = 2.5,
    *,
    mode: FanoMode | None = None,
) -> float:
    """Fano factor: variance / mean of spike counts.

    Fano = 1 for Poisson, > 1 for bursty, < 1 for regular.

    The Fano factor is well-defined only relative to a *counting unit*.
    Two counting units are supported and **they are not directly
    comparable** — pick the one that matches the question being asked
    and pin it explicitly with *mode*:

    ``mode="per_trial"`` (the standard neuroscience convention)
        For each trial in *trials*, count the spikes in that trial,
        then compute ``Var(counts) / Mean(counts)``.  *bin_size* and
        *trial_duration* are ignored — every trial counts as one
        observation regardless of how long it is.  Requires *trials*.

    ``mode="per_bin"``
        Histogram all spikes into ``trial_duration / bin_size`` bins
        of width *bin_size* and compute ``Var(counts) / Mean(counts)``
        across the bins.  This treats each bin as one observation
        and is sensitive to the chosen bin width.  *trials* is ignored.

    When *mode* is ``None`` (the deprecated default), the function
    infers the mode from *trials* (``per_trial`` if given, else
    ``per_bin``) and emits a ``DeprecationWarning``.  Pass *mode*
    explicitly to silence the warning and pin the semantics.

    Args:
        spike_times: Spike times (seconds).
        trials: Trial index per spike. Required for ``mode="per_trial"``,
            ignored for ``mode="per_bin"``.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to analyse.
        bin_size: Bin width for ``mode="per_bin"`` (seconds).
        trial_duration: Total binning window for ``mode="per_bin"``
            (seconds). Ignored for ``mode="per_trial"``.
        mode: ``"per_trial"``, ``"per_bin"``, or ``None`` (deprecated).

    Returns:
        Fano factor (float).  ``np.nan`` if the mean count is 0.

    Raises:
        ValueError: If ``mode="per_trial"`` and *trials* is ``None``,
            or if ``mode`` is not one of the supported strings.
    """
    if mode is None:
        # Backwards-compatible behaviour: infer from ``trials`` and
        # warn so callers move to the explicit form.
        inferred = "per_trial" if trials is not None else "per_bin"
        warnings.warn(
            f"fano_factor() inferred mode={inferred!r} from the presence of "
            "`trials`; this implicit behaviour is deprecated. Pass "
            "mode='per_trial' or mode='per_bin' explicitly — they "
            "are *not* directly comparable.",
            DeprecationWarning,
            stacklevel=2,
        )
        mode = inferred  # type: ignore[assignment]

    if mode not in ("per_trial", "per_bin"):
        raise ValueError(f"mode must be 'per_trial' or 'per_bin', got {mode!r}.")
    if mode == "per_trial" and trials is None:
        raise ValueError("fano_factor(mode='per_trial') requires the `trials` array.")

    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        if trials is not None:
            trials = trials[mask]

    if mode == "per_trial":
        unique_trials = np.unique(trials)
        counts = np.array(
            [np.sum(trials == t) for t in unique_trials],
            dtype=np.float64,
        )
    else:  # per_bin
        if len(spike_times) == 0:
            return np.nan
        n_bins = int(np.ceil(trial_duration / bin_size))
        edges = np.linspace(0.0, trial_duration, n_bins + 1)
        counts = np.histogram(spike_times, bins=edges)[0].astype(np.float64)

    m = counts.mean()
    if m == 0:
        return np.nan
    return float(counts.var() / m)


# ---------------------------------------------------------------------------
# Local variation (LV) — standalone
# ---------------------------------------------------------------------------


def local_variation(
    spike_times: npt.NDArray,
    trials: npt.NDArray | None = None,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
) -> float:
    """Local variation (LV) of inter-spike intervals.

    ``LV = (3/(n-1)) * Σ ((ISI_i − ISI_{i+1}) / (ISI_i + ISI_{i+1}))²``

    LV ≈ 0 for regular firing, ≈ 1 for Poisson, > 1 for bursty.
    More robust to rate changes than CV.

    For trial-based recordings, pass *trials* so the function pools
    consecutive-pair contributions over *within-trial* ISI sequences
    only — cross-trial spike pairs never enter the LV sum.  Without
    *trials* the function falls back to the legacy continuous-mode
    behaviour and computes LV from ``np.diff(np.sort(spike_times))``.

    Args:
        spike_times: Spike times (seconds).
        trials: Optional trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to analyse.

    Returns:
        LV (float).  ``np.nan`` if there are fewer than 2 valid pairs.
    """
    per_trial_isis = _per_trial_isis(
        spike_times,
        trials,
        cluster_labels=cluster_labels,
        cluster_id=cluster_id,
    )
    return _pooled_lv(per_trial_isis)


# ---------------------------------------------------------------------------
# CV of log-ISI
# ---------------------------------------------------------------------------


def cv_log_isi(
    spike_times: npt.NDArray,
    trials: npt.NDArray | None = None,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
) -> float:
    """Coefficient of variation of log\\ :sub:`10`\\ (ISI).

    Useful when the ISI distribution is approximately lognormal.
    The log base is **10** (matching the convention used in the
    spike-sorting / lognormal-ISI literature, e.g. Barbieri et al.,
    Hromadka et al.); a previous version used the natural logarithm,
    which inflated the metric by a factor of ``ln(10) ≈ 2.30``
    relative to published values.

    For trial-based recordings, pass *trials* so the metric is
    computed from *within-trial* ISIs only.  Without it, the function
    falls back to the legacy continuous-mode behaviour.

    Args:
        spike_times: Spike times (seconds).
        trials: Optional trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to analyse.

    Returns:
        CV of log\\ :sub:`10`\\ (ISI) (float).  ``np.nan`` if there are
        fewer than 2 positive ISIs, or when the geometric mean of the
        ISIs is so close to 1 s (i.e. ``mean(log10(ISI)) ≈ 0``) that
        the ratio is numerically undefined.
    """
    per_trial_isis = _per_trial_isis(
        spike_times,
        trials,
        cluster_labels=cluster_labels,
        cluster_id=cluster_id,
    )
    return _pooled_cv_log_isi(per_trial_isis)


# ---------------------------------------------------------------------------
# PSTH
# ---------------------------------------------------------------------------


def psth(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    bin_size: float = 0.01,
    trial_duration: float = 2.5,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Peri-stimulus time histogram.

    Pool spikes across all trials and histogram into fixed-width
    time bins.  Counts are divided by the number of trials and the
    bin width to yield a firing rate (Hz).

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to analyse.
        bin_size: Bin width (seconds, default 10 ms).
        trial_duration: Duration of one trial (seconds).

    Returns:
        ``(bin_centres, rate_hz)`` — time axis and firing rate per bin.
    """
    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        trials = trials[mask]

    # Use linspace + ceil to guarantee equal-width bins covering
    # [0, trial_duration].  np.arange(0, trial_duration + bin_size, bin_size)
    # is fragile under floating-point rounding: the last edge may be
    # slightly above or below trial_duration, producing an over- or
    # under-sized final bin whose firing-rate normalisation is biased.
    n_bins = int(np.ceil(trial_duration / bin_size))
    edges = np.linspace(0.0, trial_duration, n_bins + 1)
    counts, _ = np.histogram(spike_times, bins=edges)
    n_trials = len(np.unique(trials)) if len(trials) > 0 else 1
    rate = counts.astype(np.float64) / (n_trials * bin_size)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, rate


# ---------------------------------------------------------------------------
# Trial-to-trial reliability
# ---------------------------------------------------------------------------


def _per_trial_psths(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    edges: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Build a ``(n_trials, n_bins)`` per-trial PSTH array.

    Returns ``(unique_trials, psths)``.  Vectorised: spikes are
    histogrammed once with two-dimensional bin edges.
    """
    unique_trials = np.unique(trials)
    psths = np.empty((len(unique_trials), len(edges) - 1), dtype=np.float64)
    for i, t in enumerate(unique_trials):
        psths[i] = np.histogram(spike_times[trials == t], bins=edges)[0]
    return unique_trials, psths


def trial_to_trial_reliability(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    stat: str = "psth",
    bin_size: float = 0.01,
    trial_duration: float = 2.5,
    stim_frequency: float | None = None,
) -> float:
    """Trial-to-trial reliability via correlation against the mean PSTH.

    For ``stat="psth"`` (default), compute a per-trial PSTH and return
    the mean Pearson *r* of each trial against the across-trial mean
    PSTH.  This is the standard "reliability index" used in the visual
    cortex literature: it is :math:`O(n_\\text{trials})` instead of
    :math:`O(n_\\text{trials}^2)` and is interpretable as "how much of
    the average response does each trial reproduce".

    .. note::
        Earlier versions of this function returned the mean of all
        :math:`\\binom{n}{2}` pairwise correlations.  Both quantities
        rank stable cells the same way, but the mean-PSTH variant is
        the standard convention and scales linearly with the trial
        count.  If you need the full pairwise structure, use
        :func:`trial_to_trial_correlation_matrix`.

    For ``stat="f1_phase"``, compute the F1 phase for each trial's
    PSTH and return the mean resultant length :math:`|R|` of the phases
    (high means consistent phase).  Requires *stim_frequency*.

    For other *stat* values (``"mfr"``, ``"cv"``, ``"logcv"``,
    ``"lv"``, ``"lvr"``, ``"fano"``), compute the per-trial statistic
    and return ``1 / (1 + CV)`` as a consistency score (1 = perfectly
    consistent).

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to analyse.
        stat: ``"psth"`` for mean-PSTH correlation (default),
            ``"f1_phase"`` for phase consistency, or a firing-rate
            statistic name.
        bin_size: Bin width for PSTH (seconds).
        trial_duration: Trial duration (seconds).
        stim_frequency: Stimulus temporal frequency (Hz).
            Required when ``stat="f1_phase"``.

    Returns:
        Reliability score (float).  ``np.nan`` if < 2 trials.

    Raises:
        ValueError: If ``stat="f1_phase"`` and *stim_frequency*
            is ``None``.
    """
    if stat == "f1_phase" and stim_frequency is None:
        raise ValueError("stim_frequency is required when stat='f1_phase'.")
    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        trials = trials[mask]

    unique_trials = np.unique(trials)
    if len(unique_trials) < 2:
        return np.nan

    n_bins = int(np.ceil(trial_duration / bin_size))
    edges = np.linspace(0.0, trial_duration, n_bins + 1)

    if stat == "psth":
        _, psths = _per_trial_psths(spike_times, trials, edges)
        mean_psth = psths.mean(axis=0)
        if mean_psth.std() == 0:
            return np.nan
        r_values = []
        for trial_psth in psths:
            if trial_psth.std() == 0:
                continue
            r, _ = sp_stats.pearsonr(trial_psth, mean_psth)
            r_values.append(r)
        return float(np.mean(r_values)) if r_values else np.nan

    if stat == "f1_phase":
        fs = 1.0 / bin_size
        phases = []
        for t in unique_trials:
            t_spikes = spike_times[trials == t]
            counts, _ = np.histogram(t_spikes, bins=edges)
            p = _f1_phase(counts.astype(np.float64), fs, stim_frequency)
            phases.append(float(p))
        phases_arr = np.array(phases)
        # Mean resultant length |R| ∈ [0, 1].  High = consistent phase.
        return float(np.abs(np.mean(np.exp(1j * phases_arr))))

    # For other stats, compute per-trial and return consistency
    per_trial = []
    for t in unique_trials:
        t_spikes = spike_times[trials == t]
        isis = _positive_isis(t_spikes)
        dur = trial_duration

        if stat == "mfr":
            per_trial.append(len(t_spikes) / dur if dur > 0 else 0.0)
        elif stat == "cv":
            per_trial.append(float(np.std(isis) / np.mean(isis)) if len(isis) > 1 else np.nan)
        elif stat == "logcv":
            if len(isis) > 1:
                # log10 to match ``cv_log_isi`` (literature convention).
                log_i = np.log10(isis[isis > 0])
                m = np.mean(log_i)
                per_trial.append(float(np.std(log_i) / abs(m)) if abs(m) > 1e-12 else np.nan)
            else:
                per_trial.append(np.nan)
        elif stat == "lv":
            per_trial.append(_compute_lv(isis))
        elif stat == "lvr":
            per_trial.append(_compute_lvr(isis))
        elif stat == "fano":
            sub_n_bins = int(np.ceil(trial_duration / bin_size))
            sub_edges = np.linspace(0.0, trial_duration, sub_n_bins + 1)
            c, _ = np.histogram(t_spikes, bins=sub_edges)
            m = c.mean()
            per_trial.append(float(c.var() / m) if m > 0 else np.nan)
        else:
            raise ValueError(f"Unknown stat: {stat!r}")

    arr = np.array(per_trial, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return np.nan
    cv_val = float(np.std(valid) / np.abs(np.mean(valid))) if np.mean(valid) != 0 else np.nan
    return 1.0 / (1.0 + cv_val) if not np.isnan(cv_val) else np.nan


def trial_to_trial_correlation_matrix(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    bin_size: float = 0.01,
    trial_duration: float = 2.5,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Full ``(n_trials, n_trials)`` PSTH correlation matrix.

    Returns the Pearson correlation between every pair of per-trial
    PSTHs.  This is the dense version of what
    :func:`trial_to_trial_reliability` previously returned (mean over
    the upper triangle).  Use this when you want to inspect the
    structure of trial reliability — for example, to detect block-wise
    drift or sub-populations of trials with different responses.

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to analyse.
        bin_size: Bin width for PSTH (seconds).
        trial_duration: Trial duration (seconds).

    Returns:
        ``(unique_trials, corr_matrix)`` — *unique_trials* is the
        sorted unique trial-index array; *corr_matrix* is symmetric
        with ``1.0`` on the diagonal.  Trials with zero variance are
        marked with ``nan`` in their row/column.
    """
    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        trials = trials[mask]

    n_bins = int(np.ceil(trial_duration / bin_size))
    edges = np.linspace(0.0, trial_duration, n_bins + 1)
    unique_trials, psths = _per_trial_psths(spike_times, trials, edges)
    n = len(unique_trials)
    if n < 2:
        return unique_trials, np.full((n, n), np.nan, dtype=np.float64)

    # Mark zero-variance rows
    stds = psths.std(axis=1)
    nz = stds > 0

    corr = np.full((n, n), np.nan, dtype=np.float64)
    if nz.sum() < 2:
        return unique_trials, corr
    # Vectorised Pearson via standardisation + dot product
    means = psths.mean(axis=1, keepdims=True)
    centred = psths - means
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    # Avoid division-by-zero on zero-variance rows
    safe = np.where(norms > 0, norms, 1.0)
    standardised = centred / safe
    sub = standardised[nz]
    sub_corr = sub @ sub.T
    # Place into the full matrix at the non-zero indices
    nz_idx = np.where(nz)[0]
    corr[np.ix_(nz_idx, nz_idx)] = sub_corr
    # Diagonal of valid rows = 1.0 by construction
    return unique_trials, corr


# ---------------------------------------------------------------------------
# First spike latency
# ---------------------------------------------------------------------------


def first_spike_latency(
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    cluster_labels: npt.NDArray | None = None,
    cluster_id: int | None = None,
    stim_onset: float = 0.5,
) -> dict[str, float | npt.NDArray]:
    """First spike latency after stimulus onset per trial.

    Args:
        spike_times: Trial-relative spike times (seconds).
        trials: Trial index per spike.
        cluster_labels: Optional cluster filtering.
        cluster_id: Cluster to analyse.
        stim_onset: Stimulus onset time within trial (seconds).

    Returns:
        Dict with:
            ``"latencies"`` — array of first-spike latencies per trial
            (``np.nan`` for trials with no post-stimulus spike).
            ``"mean"`` — mean latency.
            ``"median"`` — median latency.
            ``"std"`` — std of latencies.
            ``"frac_responsive"`` — fraction of trials with ≥ 1 spike.
    """
    if cluster_labels is not None and cluster_id is not None:
        mask = cluster_labels == cluster_id
        spike_times = spike_times[mask]
        trials = trials[mask]

    unique_trials = np.unique(trials)
    latencies = np.full(len(unique_trials), np.nan)

    for i, t in enumerate(unique_trials):
        t_spikes = spike_times[trials == t]
        post = t_spikes[t_spikes >= stim_onset]
        if len(post) > 0:
            latencies[i] = float(post.min() - stim_onset)

    valid = latencies[~np.isnan(latencies)]
    return {
        "latencies": latencies,
        "mean": float(np.mean(valid)) if len(valid) > 0 else np.nan,
        "median": float(np.median(valid)) if len(valid) > 0 else np.nan,
        "std": float(np.std(valid)) if len(valid) > 0 else np.nan,
        "frac_responsive": float(len(valid) / len(unique_trials))
        if len(unique_trials) > 0
        else 0.0,
    }
