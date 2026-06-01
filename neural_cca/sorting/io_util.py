"""Array loader and zarr export for the spike-sorting pipeline.

This module is recording-system agnostic: build :class:`SortingData`
from raw numpy arrays via :func:`load_from_arrays`, or read a
previously exported store back via :func:`read_zarr_sorting`.

The :class:`SortingData` and :class:`SortingResult` value objects live
in :mod:`sorting.containers`; this module re-exports them so the
historical ``from sorting.io_util import SortingData`` import path
still works.

Zarr export
-----------
Two export layouts:

* **flat** — all per-spike arrays keep their original ``(n_spikes, ...)``
  shape.  Faithful to the raw data; use when downstream code already
  indexes by a flat spike index.

* **clustered** — per-spike arrays are reshaped to
  ``(n_clusters, max_spikes_per_cluster, ...)``, NaN-padded for floats
  and ``-1``-padded for integers.  Each cluster occupies one chunk,
  enabling efficient per-cluster reads.

Both layouts store the quality metrics, orientation-selectivity metrics,
and k-search results in zarr sub-groups.

``read_zarr_sorting`` reads either layout back into ``SortingResult``
and ``SortingData`` objects.

Requires the *zarr* package (``pip install neural-cca[batch]``).
"""

from __future__ import annotations

import json
import warnings
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

# Re-exported for backwards compatibility — the canonical home of these
# value objects is now ``sorting.containers``.
from .containers import SortingData, SortingResult

__all__ = [
    "SortingData",
    "SortingResult",
    "QualityMetricKind",
    "QUALITY_METRIC_KINDS",
    "load_from_arrays",
    "to_zarr_flat",
    "to_zarr_clustered",
    "read_zarr_sorting",
]


# ---------------------------------------------------------------------------
# Quality-metric schema
# ---------------------------------------------------------------------------


class QualityMetricKind(Enum):
    """Schema kind for an entry in the ``quality`` dict.

    Each entry of :attr:`SortingResult.quality` has one of these shapes.
    The zarr writer dispatches on the kind to choose the right
    serialisation strategy, and the reader uses the kind tag stored
    alongside the data to reconstruct the original Python object.

    Adding a new metric to ``evaluate_sorting`` should always be
    accompanied by an entry in :data:`QUALITY_METRIC_KINDS` so that
    serialisation has a deterministic answer for it.  Metrics that are
    not registered fall back to runtime inference (with a warning),
    which catches simple shapes but cannot guess at e.g. nested dicts.
    """

    #: A single int / float / bool, stored as a zarr group attribute.
    SCALAR = "scalar"

    #: ``dict[int, float]`` keyed by cluster ID.  Stored as a 1-D
    #: zarr array of length ``n_clusters`` with a sibling
    #: ``cluster_ids`` array, identical to the original layout.
    PER_CLUSTER_FLOAT = "per_cluster_float"

    #: ``dict[int, np.ndarray]`` — one variable-length array per
    #: cluster (e.g. a per-cluster ISI distribution).  Stored as a
    #: NaN-padded 2-D zarr array of shape ``(n_clusters, max_len)``
    #: with a sibling ``__lengths__`` attr to recover the unpadded
    #: lengths on read.
    PER_CLUSTER_ARRAY = "per_cluster_array"

    #: ``dict[int, dict[str, Any]]`` — one nested dict per cluster
    #: (e.g. a per-cluster fit-result struct).  Stored as a single
    #: JSON-encoded zarr group attribute.  Heavier than the dedicated
    #: numeric kinds but copes with arbitrary nested structures.
    NESTED_DICT = "nested_dict"


#: Central registry mapping each known ``quality`` key to its kind.
#: Update this in lockstep with :func:`sorting.sorting.evaluate_sorting`.
#: A new metric whose name is missing from this map still serialises
#: correctly via inference, but emits a warning so the omission is
#: visible in CI logs.
QUALITY_METRIC_KINDS: dict[str, QualityMetricKind] = {
    # Core scalars
    "neg_silhouette_rel": QualityMetricKind.SCALAR,
    "silhouette_mean": QualityMetricKind.SCALAR,
    "abs_rpvs": QualityMetricKind.SCALAR,
    "rel_rpvs": QualityMetricKind.SCALAR,
    "snr_weighted": QualityMetricKind.SCALAR,
    # Per-cluster floats
    "snr_per_cluster": QualityMetricKind.PER_CLUSTER_FLOAT,
    "isolation_distance": QualityMetricKind.PER_CLUSTER_FLOAT,
    "l_ratio": QualityMetricKind.PER_CLUSTER_FLOAT,
    "d_prime": QualityMetricKind.PER_CLUSTER_FLOAT,
    "peak_amplitude_snr": QualityMetricKind.PER_CLUSTER_FLOAT,
    "waveform_stability": QualityMetricKind.PER_CLUSTER_FLOAT,
    "amplitude_drift": QualityMetricKind.PER_CLUSTER_FLOAT,
    "fraction_missing": QualityMetricKind.PER_CLUSTER_FLOAT,
    # Hill 2011 contamination rate (added v0.2.0); per-cluster float
    # in [0, 0.5], NaN for clusters with fewer than 2 spikes.
    "contamination_rate_hill": QualityMetricKind.PER_CLUSTER_FLOAT,
}


def _infer_quality_metric_kind(value: Any) -> QualityMetricKind:
    """Best-effort kind inference for an unregistered quality metric.

    The writer falls back to this when a key is not in
    :data:`QUALITY_METRIC_KINDS`.  The result is heuristic and a
    ``UserWarning`` is emitted at the call site so that missing
    registry entries are visible at write time, not at read time.
    """
    if isinstance(value, (bool, int, float, np.integer, np.floating)) or value is None:
        return QualityMetricKind.SCALAR
    if isinstance(value, dict):
        if not value:
            return QualityMetricKind.PER_CLUSTER_FLOAT
        first_value = next(iter(value.values()))
        if isinstance(first_value, (bool, int, float, np.integer, np.floating)):
            return QualityMetricKind.PER_CLUSTER_FLOAT
        if isinstance(first_value, np.ndarray):
            return QualityMetricKind.PER_CLUSTER_ARRAY
        if isinstance(first_value, dict):
            return QualityMetricKind.NESTED_DICT
    raise ValueError(
        f"Cannot infer QualityMetricKind for value of type {type(value).__name__}; "
        f"register it in QUALITY_METRIC_KINDS."
    )


def load_from_arrays(
    waveforms: npt.NDArray,
    spike_times: npt.NDArray,
    trials: npt.NDArray,
    angles: npt.NDArray,
    *,
    waveform_fs: float = 32_000.0,
    n_trials: int | None = None,
    stim_window: tuple[float, float] | None = None,
    stim_frequency: float | None = None,
    metadata: dict | None = None,
) -> SortingData:
    """Build a ``SortingData`` container from raw numpy arrays.

    Use this when you already extracted waveforms and spike times
    from your recording system.

    Args:
        waveforms: (n_spikes, snippet_length) float array.
        spike_times: (n_spikes,) spike time in seconds (trial-relative).
        trials: (n_spikes,) 0-based trial index per spike.
        angles: (n_trials,) stimulus angle in degrees per trial.
        waveform_fs: Waveform sampling rate (Hz).
        n_trials: Total number of trials.  Defaults to ``len(angles)``.
        stim_window: ``(onset, end)`` of the stimulus period within
            each trial (seconds).  **Required** (no portable default).
        stim_frequency: Temporal frequency of the stimulus (Hz), or
            ``None`` (default) to disable F0/F1/F2 harmonic analysis.
        metadata: Arbitrary dict of extra information.

    Returns:
        ``SortingData`` container.

    Raises:
        ValueError: If waveforms is not 2-D, or if stim_window is None.
    """
    waveforms = np.asarray(waveforms, dtype=np.float64)
    if waveforms.ndim != 2:
        raise ValueError(f"waveforms must be 2-D, got {waveforms.ndim}-D.")
    return SortingData(
        waveforms=waveforms,
        spike_times=np.asarray(spike_times, dtype=np.float64),
        trials=np.asarray(trials, dtype=np.int64),
        angles=np.asarray(angles, dtype=np.float64),
        waveform_fs=waveform_fs,
        n_trials=n_trials,
        stim_window=stim_window,
        stim_frequency=stim_frequency,
        metadata=metadata or {},
    )


# ===================================================================
# Zarr export / import
# ===================================================================

# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------

_ZARR_MAJOR: int = 0


def _require_zarr() -> ModuleType:
    """Import *zarr*, raising a helpful ``ImportError`` on failure."""
    global _ZARR_MAJOR
    try:
        import zarr

        _ZARR_MAJOR = int(zarr.__version__.split(".")[0])
        return zarr
    except ImportError:
        raise ImportError(
            "zarr is required for zarr export/import. "
            "Install it with:  pip install neural-cca[batch]"
        ) from None


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to JSON-safe types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v):
            return None
        return v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------


def _chunks_1d(n: int, target: int = 65_536) -> tuple[int]:
    return (min(n, target),)


def _chunks_2d(shape: tuple[int, int], target_rows: int = 4096) -> tuple[int, int]:
    return (min(shape[0], target_rows), shape[1])


def _chunks_3d(
    shape: tuple[int, int, int],
    target_rows: int = 4096,
) -> tuple[int, int, int]:
    return (1, min(shape[1], target_rows), shape[2])


# ---------------------------------------------------------------------------
# Zarr v2 / v3 compatibility
# ---------------------------------------------------------------------------


def _zarr_array(
    parent: Any,
    name: str,
    data: npt.NDArray,
    chunks: tuple,
    **kw: Any,
) -> Any:
    """Create a zarr array, compatible with both zarr v2 and v3.

    Zarr v3 renamed the per-array compressor argument from
    ``compressor`` (singular, one codec) to ``compressors`` (plural,
    sequence of codecs).  The public ``compressor=`` argument on
    :func:`to_zarr_flat` / :func:`to_zarr_clustered` is translated
    here so existing callers continue to work against either zarr
    major version.  When ``compressor`` is ``None`` (the common case)
    no kwarg is forwarded and the zarr default is used.
    """
    if _ZARR_MAJOR >= 3:
        if "compressor" in kw:
            comp = kw.pop("compressor")
            # Drop the kwarg entirely when ``None`` so zarr v3 picks
            # its default codec list; wrap a single codec in a tuple
            # for the new ``compressors`` parameter.
            if comp is not None:
                kw["compressors"] = (comp,)
        return parent.create_array(
            name,
            data=data,
            chunks=chunks,
            **kw,
        )
    return parent.create_dataset(
        name,
        data=data,
        chunks=chunks,
        **kw,
    )


def _read_array(group: Any, name: str) -> npt.NDArray:
    """Read a numpy array from a zarr group member (v2/v3 safe).

    In zarr v3, ``group[name]`` may return a ``Group`` instead of an
    ``Array`` in certain edge cases.  This helper accesses ``.get()``
    when available and falls back to ``group[name][:]``, always
    returning a numpy array.
    """
    member = group[name]
    # zarr v3 Array exposes .get / numpy conversion
    if hasattr(member, "shape") and hasattr(member, "dtype"):
        return np.asarray(member[:])
    # Unexpected type — try direct read anyway (will error clearly)
    return np.asarray(member[:])


def _iter_array_names(group: Any) -> list[str]:
    """Return names of *array* members of a zarr group (v2/v3 safe).

    Skips sub-groups so that callers can safely do
    ``_read_array(group, name)`` for every returned name.
    """
    names: list[str] = []
    for name in group:
        member = group[name]
        # In zarr v2 this is zarr.core.Array; in v3 zarr.core.array.Array.
        # Both have .shape and .dtype; Groups do not expose .dtype.
        if hasattr(member, "dtype"):
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Write helpers — metadata
# ---------------------------------------------------------------------------


def _write_root_attrs(
    root: Any,
    result: Any,
    data: SortingData,
    export_mode: str,
) -> None:
    root.attrs["export_mode"] = export_mode
    root.attrs["n_clusters"] = int(result.n_clusters)
    root.attrs["n_spikes"] = int(data.n_spikes)
    root.attrs["snippet_length"] = int(data.snippet_length)
    root.attrs["waveform_fs"] = float(data.waveform_fs)
    root.attrs["n_trials"] = int(data.n_trials)
    root.attrs["stim_window"] = [float(data.stim_window[0]), float(data.stim_window[1])]
    root.attrs["stim_frequency"] = (
        float(data.stim_frequency) if data.stim_frequency is not None else None
    )
    root.attrs["sorting_metadata"] = _make_json_safe(result.metadata)
    root.attrs["data_metadata"] = _make_json_safe(data.metadata)


# ---------------------------------------------------------------------------
# Write helpers — quality
# ---------------------------------------------------------------------------


def _resolve_metric_kind(key: str, val: Any) -> QualityMetricKind:
    """Look up *key* in the registry, falling back to inference + warning."""
    kind = QUALITY_METRIC_KINDS.get(key)
    if kind is not None:
        return kind
    try:
        inferred = _infer_quality_metric_kind(val)
    except ValueError as exc:
        raise ValueError(
            f"Quality metric {key!r} is not registered in "
            f"QUALITY_METRIC_KINDS and its kind cannot be inferred: {exc}"
        ) from exc
    warnings.warn(
        f"Quality metric {key!r} is not registered in QUALITY_METRIC_KINDS; "
        f"inferred kind {inferred.value!r}.  Add it to the registry to "
        f"silence this warning and lock its serialisation behaviour.",
        UserWarning,
        stacklevel=3,
    )
    return inferred


def _write_quality_group(
    parent: Any,
    quality: dict,
    compressor: Any,
) -> None:
    """Serialise the ``quality`` dict using the registered metric kinds.

    The writer dispatches on :class:`QualityMetricKind` so that each
    metric is stored in a shape that round-trips losslessly.  A
    ``__metric_kinds__`` group attribute records the resolved kind for
    every key, which is what :func:`_read_quality_group` uses to
    reconstruct the original Python object — there is no schema
    inference at read time.
    """
    qg = parent.create_group("quality")
    scalar_attrs: dict[str, Any] = {}
    nested_attrs: dict[str, str] = {}
    array_lengths: dict[str, list[int]] = {}
    cluster_ids: list[int] | None = None
    kind_tags: dict[str, str] = {}
    kw = {"compressor": compressor} if compressor is not None else {}

    for key, val in quality.items():
        kind = _resolve_metric_kind(key, val)
        kind_tags[key] = kind.value

        if kind is QualityMetricKind.SCALAR:
            scalar_attrs[key] = _make_json_safe(val)

        elif kind is QualityMetricKind.PER_CLUSTER_FLOAT:
            ids = sorted(val.keys())
            if cluster_ids is None:
                cluster_ids = ids
            arr = np.array(
                [float(val.get(cid, np.nan)) for cid in ids],
                dtype=np.float64,
            )
            _zarr_array(qg, key, arr, (len(arr),), **kw)

        elif kind is QualityMetricKind.PER_CLUSTER_ARRAY:
            ids = sorted(val.keys())
            if cluster_ids is None:
                cluster_ids = ids
            arrays = [np.asarray(val[cid], dtype=np.float64) for cid in ids]
            max_len = max((len(a) for a in arrays), default=0)
            padded = np.full((len(ids), max_len), np.nan, dtype=np.float64)
            for i, a in enumerate(arrays):
                padded[i, : len(a)] = a
            _zarr_array(qg, key, padded, padded.shape, **kw)
            array_lengths[key] = [len(a) for a in arrays]

        elif kind is QualityMetricKind.NESTED_DICT:
            nested_attrs[key] = json.dumps(_make_json_safe(val))

        else:  # pragma: no cover - exhaustiveness guard
            raise ValueError(f"Unhandled QualityMetricKind: {kind!r}")

    if cluster_ids is not None:
        cids_arr = np.array(cluster_ids, dtype=np.int64)
        _zarr_array(qg, "cluster_ids", cids_arr, (len(cids_arr),), **kw)

    qg.attrs.update(scalar_attrs)
    qg.attrs["__metric_kinds__"] = kind_tags
    if nested_attrs:
        qg.attrs["__nested_dicts__"] = nested_attrs
    if array_lengths:
        qg.attrs["__array_lengths__"] = array_lengths


def _read_quality_group(qg: Any) -> dict:
    """Reconstruct the ``quality`` dict using the stored metric kinds.

    Falls back to the legacy inference path (no ``__metric_kinds__``
    attribute) for stores written before the schema was added.  Anyone
    on the new writer always gets the explicit dispatch.
    """
    attrs = dict(qg.attrs)
    quality: dict[str, Any] = {}

    kind_tags = attrs.pop("__metric_kinds__", None)
    nested_dicts = attrs.pop("__nested_dicts__", {}) or {}
    array_lengths = attrs.pop("__array_lengths__", {}) or {}

    if kind_tags is None:
        # Legacy path — old store written before the schema existed.
        # Use the same heuristic as the old implementation.
        for k, v in attrs.items():
            quality[k] = v
        if "cluster_ids" in qg:
            cids = _read_array(qg, "cluster_ids")
            for name in _iter_array_names(qg):
                if name == "cluster_ids":
                    continue
                arr = _read_array(qg, name)
                quality[name] = {int(cids[i]): float(arr[i]) for i in range(len(cids))}
        return quality

    # Schema-aware path.
    cids: np.ndarray | None = None
    if "cluster_ids" in qg:
        cids = _read_array(qg, "cluster_ids")

    for key, kind_str in kind_tags.items():
        kind = QualityMetricKind(kind_str)

        if kind is QualityMetricKind.SCALAR:
            quality[key] = attrs.get(key)

        elif kind is QualityMetricKind.PER_CLUSTER_FLOAT:
            if cids is None:
                raise ValueError(
                    f"Quality store missing 'cluster_ids' for per-cluster metric {key!r}."
                )
            arr = _read_array(qg, key)
            quality[key] = {int(cids[i]): float(arr[i]) for i in range(len(cids))}

        elif kind is QualityMetricKind.PER_CLUSTER_ARRAY:
            if cids is None:
                raise ValueError(
                    f"Quality store missing 'cluster_ids' for per-cluster metric {key!r}."
                )
            padded = _read_array(qg, key)
            lens = array_lengths.get(key, [padded.shape[1]] * len(cids))
            quality[key] = {
                int(cids[i]): np.asarray(padded[i, : int(lens[i])]) for i in range(len(cids))
            }

        elif kind is QualityMetricKind.NESTED_DICT:
            raw = nested_dicts.get(key)
            if raw is None:
                quality[key] = {}
            else:
                decoded = json.loads(raw)
                # JSON object keys are strings; restore int cluster IDs
                # when they parse cleanly so the structure mirrors what
                # ``evaluate_sorting`` originally produced.
                quality[key] = {
                    (int(k) if k.lstrip("-").isdigit() else k): v for k, v in decoded.items()
                }

        else:  # pragma: no cover - exhaustiveness guard
            raise ValueError(f"Unhandled QualityMetricKind: {kind!r}")

    return quality


# ---------------------------------------------------------------------------
# Write helpers — os_metrics
# ---------------------------------------------------------------------------


def _write_os_metrics_group(
    parent: Any,
    os_metrics: dict[int, dict],
    compressor: Any,
) -> None:
    og = parent.create_group("os_metrics")
    kw = {"compressor": compressor} if compressor is not None else {}

    cluster_ids = sorted(os_metrics.keys())
    cids_arr = np.array(cluster_ids, dtype=np.int64)
    _zarr_array(og, "cluster_ids", cids_arr, (len(cids_arr),), **kw)

    # Collect all metric names across clusters
    all_keys: set[str] = set()
    for metrics in os_metrics.values():
        all_keys.update(metrics.keys())

    scalar_metrics: dict[str, list[float]] = {}
    non_scalar: dict[str, dict[str, Any]] = {}

    for key in sorted(all_keys):
        values = [os_metrics[cid].get(key) for cid in cluster_ids]
        if all(isinstance(v, (int, float, np.integer, np.floating, type(None))) for v in values):
            scalar_metrics[key] = [float(v) if v is not None else np.nan for v in values]
        else:
            for cid, v in zip(cluster_ids, values):
                non_scalar.setdefault(str(cid), {})[key] = _make_json_safe(v)

    for key, vals in scalar_metrics.items():
        arr = np.array(vals, dtype=np.float64)
        _zarr_array(og, key, arr, (len(arr),), **kw)

    if non_scalar:
        og.attrs["non_scalar_json"] = json.dumps(non_scalar)


def _read_os_metrics_group(og: Any) -> dict[int, dict]:
    cids = _read_array(og, "cluster_ids")
    n = len(cids)

    result: dict[int, dict] = {int(cids[i]): {} for i in range(n)}

    # Scalar arrays
    for name in _iter_array_names(og):
        if name == "cluster_ids":
            continue
        arr = _read_array(og, name)
        for i in range(n):
            result[int(cids[i])][name] = float(arr[i])

    # Non-scalar JSON fallback
    if "non_scalar_json" in og.attrs:
        non_scalar = json.loads(og.attrs["non_scalar_json"])
        for cid_str, metrics in non_scalar.items():
            cid = int(cid_str)
            if cid in result:
                result[cid].update(metrics)

    return result


# ---------------------------------------------------------------------------
# Write helpers — k_search
# ---------------------------------------------------------------------------


def _write_k_search_group(
    parent: Any,
    k_search: dict[int, float],
    compressor: Any,
) -> None:
    kg = parent.create_group("k_search")
    kw = {"compressor": compressor} if compressor is not None else {}
    ks = sorted(k_search.keys())
    k_arr = np.array(ks, dtype=np.int64)
    s_arr = np.array([k_search[k] for k in ks], dtype=np.float64)
    _zarr_array(kg, "k_values", k_arr, (len(k_arr),), **kw)
    _zarr_array(kg, "silhouette_scores", s_arr, (len(s_arr),), **kw)


def _read_k_search_group(kg: Any) -> dict[int, float]:
    ks = _read_array(kg, "k_values")
    scores = _read_array(kg, "silhouette_scores")
    return {int(ks[i]): float(scores[i]) for i in range(len(ks))}


# ===================================================================
# Public zarr API
# ===================================================================


def to_zarr_flat(
    result: Any,
    data: SortingData,
    store: str | Path,
    *,
    overwrite: bool = False,
    compressor: object | None = None,
) -> Any:
    """Export sorting result and input data as a flat zarr store.

    All per-spike arrays retain their original ``(n_spikes, ...)``
    shape.  This is the most faithful representation of the raw data.

    Args:
        result: Output of the sorting pipeline (``SortingResult``).
        data: Input data used for sorting.
        store: Path to the zarr store directory.
        overwrite: If ``True``, overwrite an existing store.
        compressor: Zarr compressor instance (default: zarr default).

    Returns:
        The root ``zarr.Group``.

    Raises:
        ImportError: If *zarr* is not installed.
    """
    zarr = _require_zarr()
    mode = "w" if overwrite else "w-"
    root = zarr.open_group(str(store), mode=mode)

    _write_root_attrs(root, result, data, export_mode="flat")

    kw = {"compressor": compressor} if compressor is not None else {}
    n = data.n_spikes

    _zarr_array(
        root,
        "cluster_labels",
        result.cluster_labels,
        _chunks_1d(n),
        **kw,
    )
    _zarr_array(root, "spike_times", data.spike_times, _chunks_1d(n), **kw)
    _zarr_array(
        root,
        "waveforms",
        data.waveforms,
        _chunks_2d(data.waveforms.shape),
        **kw,
    )
    _zarr_array(root, "trials", data.trials, _chunks_1d(n), **kw)
    _zarr_array(
        root,
        "angles",
        data.angles,
        _chunks_1d(len(data.angles)),
        **kw,
    )

    _write_quality_group(root, result.quality, compressor)
    if result.os_metrics is not None:
        _write_os_metrics_group(root, result.os_metrics, compressor)
    if result.k_search is not None:
        _write_k_search_group(root, result.k_search, compressor)

    return root


def to_zarr_clustered(
    result: Any,
    data: SortingData,
    store: str | Path,
    *,
    overwrite: bool = False,
    compressor: object | None = None,
    fill_value_int: int = -1,
) -> Any:
    """Export sorting result reshaped by cluster membership.

    Per-spike arrays are reshaped to
    ``(n_clusters, max_spikes_per_cluster, ...)`` with ``NaN`` padding
    for float arrays and *fill_value_int* for integer arrays.  Each
    cluster occupies its own chunk along axis 0, enabling efficient
    per-cluster reads.

    A ``spike_count`` array of shape ``(n_clusters,)`` stores the
    actual (un-padded) spike count per cluster for reconstruction.
    An ``original_index`` array (same padded shape as ``spike_times``)
    records the original chronological row index of every spike, so
    that :func:`read_zarr_sorting` can undo the cluster reordering
    and return arrays in the same row order they were written in.
    Without this, the clustered round-trip would be order-changing,
    which silently breaks any code that aligns spike-indexed data
    with another array carried alongside.

    Args:
        result: Output of the sorting pipeline (``SortingResult``).
        data: Input data used for sorting.
        store: Path to the zarr store directory.
        overwrite: If ``True``, overwrite an existing store.
        compressor: Zarr compressor instance (default: zarr default).
        fill_value_int: Fill value for padded integer arrays
            (default ``-1``).

    Returns:
        The root ``zarr.Group``.

    Raises:
        ImportError: If *zarr* is not installed.
    """
    zarr = _require_zarr()
    mode = "w" if overwrite else "w-"
    root = zarr.open_group(str(store), mode=mode)

    _write_root_attrs(root, result, data, export_mode="clustered")

    kw = {"compressor": compressor} if compressor is not None else {}

    # --- Compute cluster layout ---
    cluster_ids = np.unique(result.cluster_labels)
    n_clusters = len(cluster_ids)
    counts = np.array(
        [int(np.sum(result.cluster_labels == cid)) for cid in cluster_ids],
        dtype=np.int64,
    )
    max_spk = int(counts.max())
    snip = data.snippet_length

    root.attrs["max_spikes_per_cluster"] = max_spk
    root.attrs["fill_value_int"] = fill_value_int

    # --- Build padded arrays ---
    st_pad = np.full((n_clusters, max_spk), np.nan, dtype=np.float64)
    wv_pad = np.full(
        (n_clusters, max_spk, snip),
        np.nan,
        dtype=np.float64,
    )
    tr_pad = np.full(
        (n_clusters, max_spk),
        fill_value_int,
        dtype=np.int64,
    )
    # Original (chronological) row index of each spike in the flat
    # input arrays.  Storing this lets the reader undo the cluster
    # reordering and reconstruct the exact original row order, which
    # matters whenever a downstream consumer aligns spike-indexed data
    # with another array (e.g. an external annotation column).  Without
    # it, the clustered → flat round-trip is order-changing.
    orig_idx_pad = np.full(
        (n_clusters, max_spk),
        fill_value_int,
        dtype=np.int64,
    )

    for i, cid in enumerate(cluster_ids):
        mask = result.cluster_labels == cid
        n_c = int(mask.sum())
        st_pad[i, :n_c] = data.spike_times[mask]
        wv_pad[i, :n_c, :] = data.waveforms[mask]
        tr_pad[i, :n_c] = data.trials[mask]
        orig_idx_pad[i, :n_c] = np.where(mask)[0]

    # --- Write arrays ---
    _zarr_array(
        root,
        "cluster_labels",
        result.cluster_labels,
        _chunks_1d(len(result.cluster_labels)),
        **kw,
    )
    _zarr_array(
        root,
        "cluster_ids",
        cluster_ids.astype(np.int64),
        _chunks_1d(n_clusters),
        **kw,
    )
    _zarr_array(root, "spike_count", counts, _chunks_1d(n_clusters), **kw)
    _zarr_array(
        root,
        "spike_times",
        st_pad,
        (1, min(max_spk, 65_536)),
        **kw,
    )
    _zarr_array(
        root,
        "waveforms",
        wv_pad,
        _chunks_3d(wv_pad.shape),
        **kw,
    )
    _zarr_array(
        root,
        "trials",
        tr_pad,
        (1, min(max_spk, 65_536)),
        **kw,
    )
    _zarr_array(
        root,
        "original_index",
        orig_idx_pad,
        (1, min(max_spk, 65_536)),
        **kw,
    )
    _zarr_array(
        root,
        "angles",
        data.angles,
        _chunks_1d(len(data.angles)),
        **kw,
    )

    _write_quality_group(root, result.quality, compressor)
    if result.os_metrics is not None:
        _write_os_metrics_group(root, result.os_metrics, compressor)
    if result.k_search is not None:
        _write_k_search_group(root, result.k_search, compressor)

    return root


def read_zarr_sorting(
    store: str | Path,
) -> tuple:
    """Read a sorting zarr store back into Python objects.

    Works with stores written by either :func:`to_zarr_flat` or
    :func:`to_zarr_clustered`.  For the clustered layout the padded
    arrays are automatically unpacked into their flat originals.

    Args:
        store: Path to the zarr store directory.

    Returns:
        ``(SortingResult, SortingData)`` tuple.

    Raises:
        ImportError: If *zarr* is not installed.
        ValueError: If the store has an unrecognised ``export_mode``.
    """
    zarr = _require_zarr()
    root = zarr.open_group(str(store), mode="r")

    export_mode = root.attrs["export_mode"]
    attrs = dict(root.attrs)

    # --- Read quality / os_metrics / k_search ---
    quality = _read_quality_group(root["quality"]) if "quality" in root else {}
    os_metrics = _read_os_metrics_group(root["os_metrics"]) if "os_metrics" in root else None
    k_search = _read_k_search_group(root["k_search"]) if "k_search" in root else None

    angles = _read_array(root, "angles")

    if export_mode == "flat":
        cluster_labels = _read_array(root, "cluster_labels")
        spike_times = _read_array(root, "spike_times")
        waveforms = _read_array(root, "waveforms")
        trials = _read_array(root, "trials")

    elif export_mode == "clustered":
        cluster_ids = _read_array(root, "cluster_ids")
        spike_count = _read_array(root, "spike_count")
        st_pad = _read_array(root, "spike_times")
        wv_pad = _read_array(root, "waveforms")
        tr_pad = _read_array(root, "trials")
        # Stores written before the order-preserving fix have no
        # ``original_index`` array.  Treat them as cluster-ordered (the
        # old behaviour) so old stores can still be read; new stores
        # round-trip in chronological order.
        has_orig_idx = "original_index" in root
        if has_orig_idx:
            orig_idx_pad = _read_array(root, "original_index")

        # Unpad into flat arrays — data comes out in cluster order.
        st_parts: list[npt.NDArray] = []
        wv_parts: list[npt.NDArray] = []
        tr_parts: list[npt.NDArray] = []
        lbl_parts: list[npt.NDArray] = []
        orig_parts: list[npt.NDArray] = []
        for i in range(len(cluster_ids)):
            n_c = int(spike_count[i])
            st_parts.append(st_pad[i, :n_c])
            wv_parts.append(wv_pad[i, :n_c, :])
            tr_parts.append(tr_pad[i, :n_c])
            lbl_parts.append(np.full(n_c, cluster_ids[i], dtype=np.int64))
            if has_orig_idx:
                orig_parts.append(orig_idx_pad[i, :n_c])

        spike_times = np.concatenate(st_parts)
        waveforms = np.concatenate(wv_parts)
        trials = np.concatenate(tr_parts)
        cluster_labels = np.concatenate(lbl_parts)

        if has_orig_idx:
            # Restore the original (chronological) row order so the
            # round-trip is identity-preserving.  Without this step,
            # spike_times[i] no longer corresponds to the same row in
            # any externally-aligned array the caller may carry along.
            order = np.argsort(np.concatenate(orig_parts))
            spike_times = spike_times[order]
            waveforms = waveforms[order]
            trials = trials[order]
            cluster_labels = cluster_labels[order]

    else:
        raise ValueError(f"Unrecognised export_mode in zarr store: {export_mode!r}")

    # --- Reconstruct SortingData ---
    sw = attrs["stim_window"]
    sorting_data = SortingData(
        waveforms=waveforms.astype(np.float64),
        spike_times=spike_times.astype(np.float64),
        trials=trials.astype(np.int64),
        angles=angles.astype(np.float64),
        waveform_fs=float(attrs["waveform_fs"]),
        n_trials=int(attrs["n_trials"]),
        stim_window=(float(sw[0]), float(sw[1])),
        stim_frequency=(
            float(attrs["stim_frequency"]) if attrs.get("stim_frequency") is not None else None
        ),
        metadata=attrs.get("data_metadata", {}),
    )

    # --- Reconstruct SortingResult ---
    sorting_result = SortingResult(
        cluster_labels=cluster_labels.astype(np.int64),
        n_clusters=int(attrs["n_clusters"]),
        quality=quality,
        os_metrics=os_metrics,
        k_search=k_search,
        metadata=attrs.get("sorting_metadata", {}),
    )

    return sorting_result, sorting_data
