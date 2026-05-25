"""Tests for zarr export/import of sorting results."""

from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from neural_cca.sorting.io_util import SortingData
from neural_cca.sorting.sorting import SortingResult
from neural_cca.sorting.io_util import (
    QUALITY_METRIC_KINDS,
    QualityMetricKind,
    _infer_quality_metric_kind,
    read_zarr_sorting,
    to_zarr_clustered,
    to_zarr_flat,
)


# ---------------------------------------------------------------------------
# Fixtures — the underlying data lives in conftest.py (``sample_zarr_data``)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_data(sample_zarr_data):
    """Alias for ``sample_zarr_data`` (keeps every test arg name unchanged)."""
    return sample_zarr_data


# ---------------------------------------------------------------------------
# Flat export roundtrip
# ---------------------------------------------------------------------------

class TestFlatExport:
    def test_roundtrip_labels(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        result2, data2 = read_zarr_sorting(store)
        np.testing.assert_array_equal(result.cluster_labels, result2.cluster_labels)

    def test_roundtrip_spike_times(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        _, data2 = read_zarr_sorting(store)
        np.testing.assert_allclose(data.spike_times, data2.spike_times)

    def test_roundtrip_waveforms(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        _, data2 = read_zarr_sorting(store)
        np.testing.assert_allclose(data.waveforms, data2.waveforms)

    def test_roundtrip_trials(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        _, data2 = read_zarr_sorting(store)
        np.testing.assert_array_equal(data.trials, data2.trials)

    def test_roundtrip_angles(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        _, data2 = read_zarr_sorting(store)
        np.testing.assert_allclose(data.angles, data2.angles)

    def test_roundtrip_scalar_attrs(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        result2, data2 = read_zarr_sorting(store)
        assert result2.n_clusters == result.n_clusters
        assert data2.waveform_fs == data.waveform_fs
        assert data2.n_trials == data.n_trials
        assert data2.stim_window == data.stim_window
        assert data2.stim_frequency == data.stim_frequency

    def test_roundtrip_quality_scalars(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        result2, _ = read_zarr_sorting(store)
        assert result2.quality["silhouette_mean"] == pytest.approx(0.42)
        assert result2.quality["neg_silhouette_rel"] == pytest.approx(0.05)

    def test_roundtrip_quality_per_cluster(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        result2, _ = read_zarr_sorting(store)
        for cid in [0, 1, 2]:
            assert result2.quality["snr_per_cluster"][cid] == pytest.approx(
                result.quality["snr_per_cluster"][cid]
            )
            assert result2.quality["isolation_distance"][cid] == pytest.approx(
                result.quality["isolation_distance"][cid]
            )

    def test_roundtrip_k_search(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        result2, _ = read_zarr_sorting(store)
        assert result2.k_search == result.k_search

    def test_roundtrip_os_metrics(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        to_zarr_flat(result, data, store)
        result2, _ = read_zarr_sorting(store)
        assert result2.os_metrics is not None
        for cid in [0, 1, 2]:
            assert result2.os_metrics[cid]["osi"] == pytest.approx(
                result.os_metrics[cid]["osi"]
            )

    def test_export_mode_attr(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "flat.zarr"
        root = to_zarr_flat(result, data, store)
        assert root.attrs["export_mode"] == "flat"


# ---------------------------------------------------------------------------
# Clustered export roundtrip
# ---------------------------------------------------------------------------

class TestClusteredExport:
    def test_roundtrip_labels(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        to_zarr_clustered(result, data, store)
        result2, _ = read_zarr_sorting(store)
        # Labels are reconstructed from cluster layout, so they come
        # back in cluster order — same *set* of counts per cluster.
        for cid in np.unique(result.cluster_labels):
            assert int(np.sum(result2.cluster_labels == cid)) == int(
                np.sum(result.cluster_labels == cid)
            )

    def test_roundtrip_preserves_original_order(self, sample_data, tmp_path):
        """Clustered round-trip must return arrays in chronological order.

        Storing the ``original_index`` array on write lets the reader
        undo the cluster reordering, so the round-trip is identity-
        preserving on every per-spike array.  This is the property
        external code relies on when it carries spike-aligned data
        alongside ``SortingData``.
        """
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        to_zarr_clustered(result, data, store)
        result2, data2 = read_zarr_sorting(store)

        np.testing.assert_allclose(data.spike_times, data2.spike_times)
        np.testing.assert_allclose(data.waveforms, data2.waveforms)
        np.testing.assert_array_equal(data.trials, data2.trials)
        np.testing.assert_array_equal(
            result.cluster_labels, result2.cluster_labels,
        )

    def test_original_index_array_written(self, sample_data, tmp_path):
        """``original_index`` must be present in the clustered store."""
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        root = to_zarr_clustered(result, data, store)
        assert "original_index" in root
        # Same padded shape as spike_times.
        assert root["original_index"].shape == root["spike_times"].shape

    def test_original_index_unpadded_is_permutation(self, sample_data, tmp_path):
        """The unpadded original_index must be a permutation of [0..n)."""
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        root = to_zarr_clustered(result, data, store)
        spike_count = root["spike_count"][:]
        orig_idx_pad = root["original_index"][:]
        # Concatenate per-cluster slices and check it covers exactly
        # 0 .. n_spikes - 1 once each.
        flat = np.concatenate(
            [orig_idx_pad[i, :int(spike_count[i])] for i in range(len(spike_count))]
        )
        np.testing.assert_array_equal(np.sort(flat), np.arange(data.n_spikes))

    def test_padded_shapes(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        root = to_zarr_clustered(result, data, store)

        cluster_ids = np.unique(result.cluster_labels)
        n_cl = len(cluster_ids)
        max_spk = max(int(np.sum(result.cluster_labels == c)) for c in cluster_ids)

        assert root["spike_times"].shape == (n_cl, max_spk)
        assert root["waveforms"].shape == (n_cl, max_spk, data.snippet_length)
        assert root["trials"].shape == (n_cl, max_spk)

    def test_spike_count_matches(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        root = to_zarr_clustered(result, data, store)

        cluster_ids = np.unique(result.cluster_labels)
        spike_count = root["spike_count"][:]
        for i, cid in enumerate(cluster_ids):
            assert spike_count[i] == int(np.sum(result.cluster_labels == cid))

    def test_nan_padding_floats(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        root = to_zarr_clustered(result, data, store)

        spike_count = root["spike_count"][:]
        st = root["spike_times"][:]
        for i in range(len(spike_count)):
            n_c = spike_count[i]
            max_spk = st.shape[1]
            if n_c < max_spk:
                assert np.all(np.isnan(st[i, n_c:]))

    def test_fill_value_int_padding(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        to_zarr_clustered(result, data, store, fill_value_int=-99)
        root = zarr.open_group(str(store), mode="r")

        spike_count = root["spike_count"][:]
        tr = root["trials"][:]
        for i in range(len(spike_count)):
            n_c = spike_count[i]
            max_spk = tr.shape[1]
            if n_c < max_spk:
                assert np.all(tr[i, n_c:] == -99)

    def test_export_mode_attr(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        root = to_zarr_clustered(result, data, store)
        assert root.attrs["export_mode"] == "clustered"

    def test_roundtrip_quality(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        to_zarr_clustered(result, data, store)
        result2, _ = read_zarr_sorting(store)
        assert result2.quality["silhouette_mean"] == pytest.approx(0.42)
        for cid in [0, 1, 2]:
            assert result2.quality["snr_per_cluster"][cid] == pytest.approx(
                result.quality["snr_per_cluster"][cid]
            )

    def test_chunk_layout(self, sample_data, tmp_path):
        """Clustered waveforms should chunk 1 cluster per chunk."""
        data, result = sample_data
        store = tmp_path / "clustered.zarr"
        root = to_zarr_clustered(result, data, store)
        assert root["waveforms"].chunks[0] == 1


# ---------------------------------------------------------------------------
# Edge cases & error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_os_metrics(self, sample_data, tmp_path):
        data, result = sample_data
        result.os_metrics = None
        store = tmp_path / "no_os.zarr"
        to_zarr_flat(result, data, store)
        result2, _ = read_zarr_sorting(store)
        assert result2.os_metrics is None

    def test_no_k_search(self, sample_data, tmp_path):
        data, result = sample_data
        result.k_search = None
        store = tmp_path / "no_ks.zarr"
        to_zarr_flat(result, data, store)
        result2, _ = read_zarr_sorting(store)
        assert result2.k_search is None

    def test_no_stim_frequency(self, sample_data, tmp_path):
        data, result = sample_data
        data.stim_frequency = None
        store = tmp_path / "no_sf.zarr"
        to_zarr_flat(result, data, store)
        _, data2 = read_zarr_sorting(store)
        assert data2.stim_frequency is None

    def test_overwrite_false_raises(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "exists.zarr"
        to_zarr_flat(result, data, store)
        with pytest.raises(Exception):
            to_zarr_flat(result, data, store, overwrite=False)

    def test_overwrite_true_succeeds(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "overwrite.zarr"
        to_zarr_flat(result, data, store)
        to_zarr_flat(result, data, store, overwrite=True)
        result2, _ = read_zarr_sorting(store)
        np.testing.assert_array_equal(result.cluster_labels, result2.cluster_labels)

    def test_single_cluster(self, tmp_path):
        """Edge case: only one cluster, no padding needed."""
        rng = np.random.default_rng(0)
        n = 50
        data = SortingData(
            waveforms=rng.standard_normal((n, 20)),
            spike_times=rng.uniform(0, 2.5, n),
            trials=rng.integers(0, 12, n).astype(np.int64),
            angles=np.linspace(0, 330, 12),
        )
        result = SortingResult(
            cluster_labels=np.zeros(n, dtype=np.int64),
            n_clusters=1,
            quality={"silhouette_mean": 1.0},
        )

        store = tmp_path / "single.zarr"
        to_zarr_clustered(result, data, store)
        result2, data2 = read_zarr_sorting(store)

        np.testing.assert_array_equal(result.cluster_labels, result2.cluster_labels)
        np.testing.assert_allclose(data.spike_times, data2.spike_times)
        np.testing.assert_allclose(data.waveforms, data2.waveforms)

    def test_metadata_roundtrip(self, sample_data, tmp_path):
        data, result = sample_data
        store = tmp_path / "meta.zarr"
        to_zarr_flat(result, data, store)
        result2, data2 = read_zarr_sorting(store)
        assert data2.metadata["electrode"] == 7
        assert data2.metadata["animal"] == "mouse_01"
        assert result2.metadata["rng"] == 0


# ---------------------------------------------------------------------------
# Quality-metric schema (QualityMetricKind + registry)
# ---------------------------------------------------------------------------

class TestQualityMetricSchema:
    """Tests for the explicit QualityMetricKind dispatch."""

    def test_registry_covers_all_evaluate_sorting_keys(self, sample_data):
        """Every key produced by ``evaluate_sorting`` must be registered.

        Catches the case where a new metric is added to
        ``evaluate_sorting`` but the author forgets to register its
        kind.  Without this guard, the writer falls back to inference
        and emits a warning instead of failing loud.
        """
        _, result = sample_data
        for key in result.quality:
            assert key in QUALITY_METRIC_KINDS, (
                f"Quality metric {key!r} is not registered in "
                f"QUALITY_METRIC_KINDS — add it."
            )

    def test_registered_kinds_match_fixture_shapes(self, sample_data):
        """Each registered kind must agree with the actual value shape."""
        _, result = sample_data
        for key, val in result.quality.items():
            kind = QUALITY_METRIC_KINDS[key]
            inferred = _infer_quality_metric_kind(val)
            assert inferred is kind, (
                f"Registered kind {kind} for {key!r} disagrees with "
                f"inferred kind {inferred} from the fixture value."
            )

    def test_kinds_attr_written_on_disk(self, sample_data, tmp_path):
        """The store carries an explicit kind tag for every metric."""
        data, result = sample_data
        store = tmp_path / "kinds.zarr"
        to_zarr_flat(result, data, store)
        root = zarr.open_group(str(store), mode="r")
        kind_tags = dict(root["quality"].attrs.get("__metric_kinds__", {}))
        assert set(kind_tags.keys()) == set(result.quality.keys())
        for key, kind in QUALITY_METRIC_KINDS.items():
            if key in kind_tags:
                assert kind_tags[key] == kind.value

    def test_unregistered_metric_emits_warning_on_write(
        self, sample_data, tmp_path,
    ):
        """An unknown metric still serialises but warns the author."""
        data, result = sample_data
        result.quality["my_custom_metric"] = 0.123  # scalar shape
        store = tmp_path / "warn.zarr"
        with pytest.warns(UserWarning, match="not registered"):
            to_zarr_flat(result, data, store)
        # And the unknown metric round-trips correctly via inference.
        result2, _ = read_zarr_sorting(store)
        assert result2.quality["my_custom_metric"] == pytest.approx(0.123)

    def test_per_cluster_array_metric_roundtrip(self, sample_data, tmp_path):
        """A PER_CLUSTER_ARRAY metric round-trips losslessly."""
        data, result = sample_data
        # Synthesise a per-cluster ISI distribution (variable length).
        isis = {
            0: np.array([0.012, 0.018, 0.022], dtype=np.float64),
            1: np.array([0.005, 0.009], dtype=np.float64),
            2: np.array([0.030, 0.040, 0.050, 0.060], dtype=np.float64),
        }
        result.quality["isi_distribution"] = isis
        # Register at runtime so the writer takes the array branch.
        QUALITY_METRIC_KINDS["isi_distribution"] = (
            QualityMetricKind.PER_CLUSTER_ARRAY
        )
        try:
            store = tmp_path / "arr.zarr"
            to_zarr_flat(result, data, store)
            result2, _ = read_zarr_sorting(store)
            for cid, arr in isis.items():
                np.testing.assert_allclose(
                    result2.quality["isi_distribution"][cid], arr,
                )
        finally:
            del QUALITY_METRIC_KINDS["isi_distribution"]

    def test_nested_dict_metric_roundtrip(self, sample_data, tmp_path):
        """A NESTED_DICT metric round-trips losslessly via JSON."""
        data, result = sample_data
        nested = {
            0: {"score": 0.8, "fano": 1.2},
            1: {"score": 0.5, "fano": 0.9},
            2: {"score": 0.95, "fano": 1.5},
        }
        result.quality["fit_summary"] = nested
        QUALITY_METRIC_KINDS["fit_summary"] = QualityMetricKind.NESTED_DICT
        try:
            store = tmp_path / "nested.zarr"
            to_zarr_flat(result, data, store)
            result2, _ = read_zarr_sorting(store)
            recovered = result2.quality["fit_summary"]
            assert set(recovered.keys()) == set(nested.keys())
            for cid in nested:
                assert recovered[cid]["score"] == pytest.approx(
                    nested[cid]["score"]
                )
                assert recovered[cid]["fano"] == pytest.approx(
                    nested[cid]["fano"]
                )
        finally:
            del QUALITY_METRIC_KINDS["fit_summary"]

    def test_string_keyed_dict_metric_no_longer_crashes(
        self, sample_data, tmp_path,
    ):
        """A dict[str, float] metric must not corrupt the writer.

        Before the schema fix, the writer would treat any dict as
        ``per_cluster_float`` and try to coerce string keys to int64,
        producing a confusing failure deep in zarr.  The schema-aware
        writer either serialises it correctly via the registry or
        raises a clear ValueError when the kind cannot be inferred.
        """
        data, result = sample_data
        # A nested-dict-shaped metric (dict[int, dict[str, ...]]) — the
        # natural shape that used to break the writer.
        result.quality["xy_centroid"] = {
            0: {"x": 0.1, "y": 0.2},
            1: {"x": 0.3, "y": 0.4},
            2: {"x": 0.5, "y": 0.6},
        }
        store = tmp_path / "xy.zarr"
        with pytest.warns(UserWarning, match="not registered"):
            to_zarr_flat(result, data, store)
        result2, _ = read_zarr_sorting(store)
        recovered = result2.quality["xy_centroid"]
        assert recovered[0]["x"] == pytest.approx(0.1)
        assert recovered[1]["y"] == pytest.approx(0.4)
