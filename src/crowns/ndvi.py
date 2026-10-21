"""NDVI from CIR imagery, resampled onto the lidar grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .config import CRS
from .tiling import Grid


def ndvi_on_grid(image_path: Path, grid: Grid, band_nir: int, band_red: int) -> np.ndarray:
    """(NIR - R) / (NIR + R), averaged into the grid cells. NaN where there is no imagery."""
    with rasterio.open(image_path) as src:
        nir = src.read(band_nir).astype(np.float32)
        red = src.read(band_red).astype(np.float32)
        total = nir + red
        ndvi = np.full(nir.shape, np.nan, dtype=np.float32)
        np.divide(nir - red, total, out=ndvi, where=total > 0)
        out = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(
            source=ndvi,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs or CRS,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=CRS,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return out
