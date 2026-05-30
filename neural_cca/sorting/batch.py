"""Batch spike sorting across all electrodes of an experiment.

The main entry point is ``batch_sort_experiment``, which loads data
from a VisionICeIO directory or zarr store, iterates over electrodes,
runs the sorting pipeline on each, computes spike-train statistics,
and writes a consolidated zarr store with per-cluster results.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .containers import SortingData
from .sorting import (
    _as_seed,
    run_sorting_pipeline,
)

__all__ = ["batch_sort_experiment"]


def batch_sort_experiment(
    data_source: str | Path,
    name: str | None = None,
    output_path: str | Path | None = None,
    electrode_indices: Sequence[int] | None = None,
    n_clusters: int | None = None,
    k_range: Sequence[int] = range(2, 6),
    tlabel2angle: dict[int, float] | None = None,
    n_angle_steps: int | None = None,
    stim_window: tuple[float, float] | None = None,
    stim_frequency: float | None = None,
    refractory_period: float = 0.001,
    compute_sta: bool = True,
    compute_tuning: bool = True,
    rng: np.random.Generator | int | None = None,
    **pipeline_kwargs,
) -> dict:
    """Batch spike sorting and analysis across all electrodes.

    Iterates through every electrode of an experiment, applies spike
    sorting, computes quality metrics, optional spike-train statistics
    (MFR, CV, LvR) and orientation selectivity per cluster, then
    writes all results into a consolidated zarr store.

    The output zarr contains:

    - ``trial_angles`` — ``(n_trials,)`` stimulus angle per trial.
    - ``spike_times_by_cluster`` — ragged, NaN-padded
      ``(n_electrodes, max_clusters, n_trials, max_spikes_per_trial)``.
    - ``firing_rate_by_trial`` —
      ``(n_electrodes, max_clusters, n_trials)``.
    - Metadata attributes: experiment info, per-electrode sorting
      quality, and per-cluster STA metrics.

    .. important::

        Three formerly-defaulted arguments are now **required** and
        raise :class:`ValueError` when omitted:

        - ``stim_window`` — there is no library default because the
          ``(onset, end)`` window is recording-specific.  Pass e.g.
          ``stim_window=(0.5, 2.5)`` for a 500 ms baseline + 2 s
          stimulus protocol.
        - ``tlabel2angle`` *or* ``n_angle_steps`` — either an explicit
          mapping or the equispaced shorthand must be supplied.  Pass
          ``tlabel2angle={1: 0.0, 2: 30.0, ...}`` or
          ``n_angle_steps=12`` (the LabView 30-degree convention).
          See ``vision_ice_analysis.steps2degree`` for the helper.

        ``stim_frequency=None`` (the new default) is permitted and
        disables F1 / F0 modulation in the per-cluster tuning block;
        pass a number to enable it.

    Args:
        data_source: Path to either a VisionICeIO experiment directory
            (with .swa/.spi/.stm/.ana files) or a zarr store
            previously created by ``visioniceio.Experiment``.
        name: Experiment name (file prefix).  Required for raw
            directories; ignored for zarr stores.
        output_path: Where to save the results zarr.  Defaults to
            ``data_source`` with suffix ``_sorted.zarr``.
        electrode_indices: Which electrodes to process.  ``None``
            processes all.
        n_clusters: Fixed cluster count per electrode.  ``None`` →
            auto-select via silhouette from *k_range*.
        k_range: Candidate k values for auto-selection.
        tlabel2angle: Mapping from 1-based stimulus label to angle.
            **Required** unless *n_angle_steps* is given; see the
            "Important" note above.
        n_angle_steps: Number of equidistant angle steps (used when
            *tlabel2angle* is ``None``).  **Required** unless
            *tlabel2angle* is given.
        stim_window: ``(onset, end)`` of the stimulus period within
            each trial (seconds).  **Required.**
        stim_frequency: Temporal frequency of the stimulus (Hz).
            ``None`` (default) disables F1/F0 computation in the
            per-cluster tuning block.
        refractory_period: Refractory period for RPV computation (s).
        compute_sta: Whether to compute spike-train statistics per
            cluster.
        compute_tuning: Whether to compute orientation selectivity per
            cluster.
        rng: Generator, int seed, or ``None`` for KMeans reproducibility.
        **pipeline_kwargs: Extra keyword arguments forwarded verbatim to
            :func:`run_sorting_pipeline` per electrode.  Use this to set
            uncommon options (``min_silhouette``, ``preprocess``,
            ``pca_components``, ``invert_waveforms``, ``bin_size``, ``n_init``)
            without inflating the batch signature.  An unrecognised key
            surfaces as a normal ``TypeError`` from the per-electrode call.

    Returns:
        Dict with keys:
            ``result_path`` — path to output zarr store.
            ``n_electrodes_processed`` — count of successful electrodes.
            ``n_clusters_total`` — total clusters across all electrodes.
            ``summary`` — per-electrode summary with quality, n_clusters,
            and optional STA / tuning metrics.

    Raises:
        ImportError: If zarr, xarray, or visioniceio are not available.
        FileNotFoundError: If *data_source* does not exist.
        ValueError: If *stim_window* is ``None``, or if both
            *tlabel2angle* and *n_angle_steps* are ``None``.
    """
    if stim_window is None:
        raise ValueError(
            "stim_window is required. Pass e.g. stim_window=(0.5, 2.5) "
            "for a 500ms baseline + 2s stimulus protocol. There is no "
            "library default because stim windows are recording-specific."
        )
    if tlabel2angle is None and n_angle_steps is None:
        raise ValueError(
            "Either tlabel2angle or n_angle_steps is required. "
            "Pass tlabel2angle={1: 0.0, 2: 30.0, ...} or n_angle_steps=12 "
            "(LabView 30-degree convention). See "
            "vision_ice_analysis.steps2degree for the helper."
        )
    # If only n_angle_steps is supplied, build tlabel2angle from it:
    if tlabel2angle is None:
        from .._utils import steps2degree

        tlabel2angle = steps2degree(n_angle_steps)

    import xarray as xr

    data_source = Path(data_source)
    if not data_source.exists():
        raise FileNotFoundError(f"Data source not found: {data_source}")

    if output_path is None:
        output_path = data_source.parent / (data_source.stem + "_sorted.zarr")
    output_path = Path(output_path)

    # ------------------------------------------------------------------
    # 1. Load experiment
    # ------------------------------------------------------------------
    if str(data_source).endswith(".zarr"):
        ds = xr.open_zarr(str(data_source))
        sample_rate_spike = ds.attrs.get("SpikeSamplingFrequency", 32_000.0)
        exp_metadata = dict(ds.attrs)
    else:
        from visioniceio.experiment import Experiment

        exp = Experiment()
        exp.load_from_dir(path=str(data_source), name=name, save_as=None)
        ds = exp.data
        sample_rate_spike = exp.sample_rate_spike
        exp_metadata = exp.metadata if exp.metadata else {}

    # ------------------------------------------------------------------
    # 2. Determine electrodes
    # ------------------------------------------------------------------
    all_electrodes = ds.electrodes.values
    if electrode_indices is not None:
        all_electrodes = np.array([e for e in electrode_indices if e in all_electrodes])

    # Stimulus angles.  Validate that ``tlabel2angle`` covers every
    # observed stimulus label up-front and raise a clear error rather
    # than letting the comprehension below blow up with a bare KeyError
    # deep in the call stack.  Matches the validation
    # ``vision_ice_analysis.load_from_visioniceio`` does at the bridge
    # layer.  When ``compute_tuning=False`` we relax the requirement
    # and fill missing labels with ``0.0`` — the angles are still
    # written into ``SortingData`` (the container always carries the
    # field) but never consumed for analysis, so the placeholder is
    # harmless.  Real coverage is still required for the tuning path
    # because ``get_os_metrics`` would otherwise alias unrelated
    # conditions to the same angle.
    observed_labels = sorted({int(lbl) for lbl in ds.stim_label.values})
    missing = [lbl for lbl in observed_labels if lbl not in tlabel2angle]
    if missing:
        if compute_tuning:
            raise KeyError(
                f"tlabel2angle does not cover stimulus labels {missing}. "
                f"Observed labels: {observed_labels}. "
                f"Mapped labels:   {sorted(tlabel2angle.keys())}. "
                f"Pass an explicit tlabel2angle covering every label "
                f"present in the recording (or disable tuning with "
                f"compute_tuning=False if angles are not meaningful for "
                f"this protocol — e.g. a mixed dot/grating-contrast "
                f"design where labels do not all correspond to one "
                f"orientation per stimulus type)."
            )
        # compute_tuning is off — fill the gaps so SortingData has the
        # full angles array but downstream never relies on them.
        tlabel2angle = {**tlabel2angle, **{lbl: 0.0 for lbl in missing}}
    angles = np.array(
        [tlabel2angle[int(lbl)] for lbl in ds.stim_label.values],
        dtype=np.float64,
    )
    n_trials = len(angles)

    # ------------------------------------------------------------------
    # 3. Process each electrode
    # ------------------------------------------------------------------
    summary: dict[int, dict] = {}
    all_electrode_results: list[dict] = []

    for elec_idx in all_electrodes:
        elec = int(elec_idx)
        try:
            # --- Extract single electrode ---
            wv_xa = ds.waveforms.sel(electrodes=elec)
            wv_stacked = wv_xa.stack(sidx=("trials", "spikes_idx"))
            wv_stacked = wv_stacked.dropna(dim="sidx", how="all")
            waveforms = wv_stacked.T.values.astype(np.float64)

            st_xa = ds.spike_times.sel(electrodes=elec)
            st_stacked = st_xa.stack(sidx=("trials", "spikes_idx"))
            st_stacked = st_stacked.dropna(dim="sidx", how="all")
            spike_times = st_stacked.values.astype(np.float64)

            trials_arr = wv_stacked.trials.values.astype(np.int64)

            if len(waveforms) < 10:
                warnings.warn(
                    f"Electrode {elec}: only {len(waveforms)} spikes, skipping.",
                    stacklevel=2,
                )
                continue

            data = SortingData(
                waveforms=waveforms,
                spike_times=spike_times,
                trials=trials_arr,
                angles=angles,
                waveform_fs=float(sample_rate_spike),
                n_trials=n_trials,
                stim_window=stim_window,
                stim_frequency=stim_frequency,
                metadata={"electrode": elec},
            )

            # --- Sort ---
            result = run_sorting_pipeline(
                data,
                n_clusters=n_clusters,
                k_range=k_range,
                rng=rng,
                refractory_period=refractory_period,
                compute_os=compute_tuning,
                plot=False,
                **pipeline_kwargs,
            )

            # --- STA per cluster ---
            sta_metrics: dict[int, dict] | None = None
            if compute_sta:
                try:
                    from ..spike_train.analysis import minimal_spike_train_analysis

                    sta_metrics = {}
                    for cl in np.unique(result.cluster_labels):
                        sta_metrics[int(cl)] = minimal_spike_train_analysis(
                            spike_times,
                            cluster_labels=result.cluster_labels,
                            cluster_id=int(cl),
                            refractory_period=refractory_period,
                            stim_window=stim_window,
                            n_trials=n_trials,
                        )
                except ImportError:
                    warnings.warn(
                        "sta not available; skipping STA metrics.",
                        stacklevel=2,
                    )

            # --- Firing rates per trial per cluster ---
            fr_by_trial: dict[int, np.ndarray] = {}
            s_on, s_end = stim_window
            stim_dur = s_end - s_on
            for cl in np.unique(result.cluster_labels):
                rates = np.zeros(n_trials, dtype=np.float64)
                cl_mask = result.cluster_labels == cl
                cl_st = spike_times[cl_mask]
                cl_tr = trials_arr[cl_mask]
                for t in range(n_trials):
                    t_spikes = cl_st[cl_tr == t]
                    rates[t] = np.sum((t_spikes > s_on) & (t_spikes <= s_end)) / stim_dur
                fr_by_trial[int(cl)] = rates

            # --- Spike times per trial per cluster ---
            st_by_trial: dict[int, dict[int, np.ndarray]] = {}
            for cl in np.unique(result.cluster_labels):
                cl_mask = result.cluster_labels == cl
                cl_st = spike_times[cl_mask]
                cl_tr = trials_arr[cl_mask]
                st_by_trial[int(cl)] = {}
                for t in range(n_trials):
                    st_by_trial[int(cl)][t] = cl_st[cl_tr == t]

            all_electrode_results.append(
                {
                    "electrode": elec,
                    "result": result,
                    "sta_metrics": sta_metrics,
                    "fr_by_trial": fr_by_trial,
                    "st_by_trial": st_by_trial,
                }
            )

            elec_summary = {
                "n_clusters": result.n_clusters,
                "quality": result.quality,
                "n_spikes": len(waveforms),
            }
            if sta_metrics:
                elec_summary["sta_metrics"] = sta_metrics
            if result.os_metrics:
                elec_summary["os_metrics"] = result.os_metrics
            summary[elec] = elec_summary

        except Exception as e:
            warnings.warn(
                f"Electrode {elec}: processing failed — {e}",
                stacklevel=2,
            )
            continue

    # ------------------------------------------------------------------
    # 4. Build and save output zarr
    # ------------------------------------------------------------------
    n_proc = len(all_electrode_results)
    if n_proc == 0:
        warnings.warn("No electrodes were successfully processed.", stacklevel=2)
        return {
            "result_path": str(output_path),
            "n_electrodes_processed": 0,
            "n_clusters_total": 0,
            "summary": summary,
        }

    max_clusters = max(r["result"].n_clusters for r in all_electrode_results)
    total_clusters = sum(r["result"].n_clusters for r in all_electrode_results)

    # Find max spikes per trial across all results
    max_spk_per_trial = 0
    for r in all_electrode_results:
        for cl_trials in r["st_by_trial"].values():
            for st_arr in cl_trials.values():
                max_spk_per_trial = max(max_spk_per_trial, len(st_arr))
    max_spk_per_trial = max(max_spk_per_trial, 1)

    processed_electrodes = np.array([r["electrode"] for r in all_electrode_results], dtype=np.int64)

    # Allocate arrays.  Use float64 throughout for numerical fidelity:
    # spike times are seconds with microsecond precision, and float32
    # only has ~7 decimal digits — converting to float32 silently loses
    # tens of microseconds for spike times near 100 s.  This also keeps
    # the dtype consistent with the io_util.to_zarr_* functions, which
    # roundtrip everything as float64.
    spike_times_arr = np.full(
        (n_proc, max_clusters, n_trials, max_spk_per_trial),
        np.nan,
        dtype=np.float64,
    )
    firing_rates_arr = np.full(
        (n_proc, max_clusters, n_trials),
        np.nan,
        dtype=np.float64,
    )
    n_clusters_arr = np.zeros(n_proc, dtype=np.int32)

    for i, r in enumerate(all_electrode_results):
        n_cl = r["result"].n_clusters
        n_clusters_arr[i] = n_cl
        for cl_idx, cl in enumerate(sorted(r["fr_by_trial"].keys())):
            firing_rates_arr[i, cl_idx, :] = r["fr_by_trial"][cl]
            for t in range(n_trials):
                st = r["st_by_trial"][cl].get(t, np.array([]))
                spike_times_arr[i, cl_idx, t, : len(st)] = st

    out_ds = xr.Dataset(
        data_vars={
            "spike_times_by_cluster": xr.DataArray(
                spike_times_arr,
                dims=("electrodes", "clusters", "trials", "spike_idx"),
                attrs={"description": "Spike times per cluster per trial, NaN-padded"},
            ),
            "firing_rate_by_trial": xr.DataArray(
                firing_rates_arr,
                dims=("electrodes", "clusters", "trials"),
                attrs={"description": "Firing rate (Hz) per cluster per trial"},
            ),
            "trial_angles": xr.DataArray(
                angles,
                dims=("trials",),
                attrs={"description": "Stimulus angle (degrees) per trial"},
            ),
            "n_clusters": xr.DataArray(
                n_clusters_arr,
                dims=("electrodes",),
                attrs={"description": "Number of clusters per electrode"},
            ),
        },
        coords={
            "electrodes": processed_electrodes,
            "clusters": np.arange(max_clusters),
            "trials": np.arange(n_trials),
            "spike_idx": np.arange(max_spk_per_trial),
        },
        attrs={
            "description": "Batch spike sorting results",
            "rng": _as_seed(rng),
            "refractory_period": refractory_period,
            "stim_window": [float(stim_window[0]), float(stim_window[1])],
            **{k: str(v) for k, v in exp_metadata.items()},
        },
    )

    out_ds.to_zarr(str(output_path), mode="w")

    return {
        "result_path": str(output_path),
        "n_electrodes_processed": n_proc,
        "n_clusters_total": total_clusters,
        "summary": summary,
    }
