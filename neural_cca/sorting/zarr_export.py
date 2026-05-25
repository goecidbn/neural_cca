"""Backwards-compatibility re-export — all zarr I/O now lives in io_util."""

from .io_util import read_zarr_sorting, to_zarr_clustered, to_zarr_flat  # noqa: F401

__all__ = ["to_zarr_flat", "to_zarr_clustered", "read_zarr_sorting"]
