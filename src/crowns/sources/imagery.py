"""CIR imagery per tile: cut from local rasters, or requested from a WMS and saved as GeoTIFF."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from shapely.geometry import box

from ..config import CRS, Config
from ..tiling import Bounds, Grid
from .http import get

RASTER_SUFFIXES = (".tif", ".tiff", ".jp2", ".vrt", ".ecw", ".png", ".jpg")


def image_for_tile(cfg: Config, bounds: Bounds, dest: Path) -> Path | None:
    """Write the imagery for `bounds` to `dest` (GeoTIFF). None when imagery is disabled or missing."""
    im = cfg.imagery
    if not im.enabled:
        return None
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if im.local_dir:
        return _from_local(cfg.path(im.local_dir), bounds, im.resolution, dest)
    return _from_wms(cfg, bounds, dest)


@lru_cache(maxsize=8)
def _local_index(folder: Path) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in RASTER_SUFFIXES)
    out = []
    for p in files:
        with rasterio.open(p) as src:
            out.append((str(p), tuple(src.bounds)))
    return tuple(out)


def _from_local(folder: Path, bounds: Bounds, res: float, dest: Path) -> Path | None:
    area = box(*bounds)
    files = [p for p, b in _local_index(folder) if box(*b).intersects(area)]
    if not files:
        return None
    sources = [rasterio.open(p) for p in files]
    try:
        data, transform = merge(sources, bounds=bounds, res=res, nodata=0)
        profile = dict(driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0],
                       dtype=data.dtype, crs=sources[0].crs or CRS, transform=transform, nodata=0,
                       compress="deflate", tiled=True)
    finally:
        for s in sources:
            s.close()
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(data)
    return dest


def _from_wms(cfg: Config, bounds: Bounds, dest: Path) -> Path:
    im = cfg.imagery
    grid = Grid.from_bounds(bounds, im.resolution)
    full: np.ndarray | None = None
    step = im.wms_max_px
    for r0 in range(0, grid.height, step):
        for c0 in range(0, grid.width, step):
            h = min(step, grid.height - r0)
            w = min(step, grid.width - c0)
            xmin = grid.xmin + c0 * im.resolution
            ymax = grid.ymax - r0 * im.resolution
            bbox = (xmin, ymax - h * im.resolution, xmin + w * im.resolution, ymax)
            block = _get_map(im.wms_url, im.wms_layer, im.wms_format, bbox, w, h)
            if full is None:
                full = np.zeros((block.shape[0], grid.height, grid.width), dtype=block.dtype)
            full[:, r0:r0 + h, c0:c0 + w] = block[:, :h, :w]
    assert full is not None
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=full.shape[0], dtype=full.dtype,
                   crs=CRS, transform=grid.transform, compress="deflate", tiled=True)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(full)
    return dest


def _get_map(url: str, layer: str, fmt: str, bbox: Bounds, width: int, height: int) -> np.ndarray:
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": layer,
        "styles": "",
        "crs": CRS,  # EPSG:28992 has easting/northing axis order, so the bbox is x/y
        "bbox": ",".join(f"{v:.3f}" for v in bbox),
        "width": width,
        "height": height,
        "format": fmt,
    }
    r = get(url, params=params)
    ctype = r.headers.get("Content-Type", "")
    if "xml" in ctype or not ctype.startswith("image"):
        raise RuntimeError(f"WMS returned {ctype or 'no content type'}: {r.text[:500]}")
    with MemoryFile(r.content) as mem, mem.open() as src:
        data = src.read()
    return data[:3]  # colour bands only (drops an alpha band if present)
