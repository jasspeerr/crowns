"""Run the detection per tile and merge the results.

Each tile runs in two stages, both cached in <workdir>/tiles/<tile_id>/:
  A. rasters: read the point cloud and imagery, build CHM, NDVI, return spread and building mask
     (slow; redone only when lidar/imagery/tiling settings change)
  B. detect: mask, tree tops, watershed, polygons
     (fast; redone automatically when mask/treetop/segment/polygon settings change)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from .config import CRS, Config
from .lidar import build_rasters, read_points
from .mask import vegetation_mask
from .ndvi import ndvi_on_grid
from .polygons import COLUMNS, crowns_to_polygons, empty_crowns
from .segment import segment_crowns
from .sources.ahn import AhnSource
from .sources.imagery import image_for_tile
from .tiling import Grid, Tile
from .treetops import find_treetops, smooth_chm

log = logging.getLogger(__name__)

RASTER_SECTIONS = ("tiling", "lidar", "imagery")
DETECT_SECTIONS = ("mask", "treetops", "segment", "polygons")


def settings_hash(cfg: Config, sections: tuple[str, ...]) -> str:
    data = {s: dataclasses.asdict(getattr(cfg, s)) for s in sections}
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


def tile_dir(cfg: Config, tile_id: str) -> Path:
    return cfg.workdir / "tiles" / tile_id


def cir_path(cfg: Config, tile_id: str) -> Path:
    """Imagery file per tile; the name changes with the imagery settings (e.g. another year)."""
    return tile_dir(cfg, tile_id) / f"cir_{settings_hash(cfg, ('imagery',))}.tif"


@dataclass
class TileRasters:
    grid: Grid
    chm: np.ndarray
    spread: np.ndarray
    building: np.ndarray
    ndvi: np.ndarray | None
    point_count: int


# -- stage A ------------------------------------------------------------------------------------

def prepare_rasters(cfg: Config, tile: Tile, ahn: AhnSource) -> TileRasters:
    d = tile_dir(cfg, tile.id)
    cache = d / "rasters.npz"
    key = settings_hash(cfg, RASTER_SECTIONS)
    grid = Grid.from_bounds(tile.buffered, cfg.lidar.resolution)
    if cache.exists() and not cfg.run.overwrite:
        with np.load(cache) as z:
            if str(z["key"]) == key:
                ndvi = z["ndvi"] if z["has_ndvi"] else None
                return TileRasters(grid, z["chm"], z["spread"], z["building"], ndvi, int(z["point_count"]))

    d.mkdir(parents=True, exist_ok=True)
    points = read_points(ahn.files_for(tile.buffered), tile.buffered)
    lr = build_rasters(points, grid, cfg.lidar)
    del points

    ndvi = None
    image = image_for_tile(cfg, tile.buffered, cir_path(cfg, tile.id))
    if image is not None:
        ndvi = ndvi_on_grid(image, grid, cfg.imagery.band_nir, cfg.imagery.band_red)

    np.savez_compressed(
        cache, key=key, chm=lr.chm.astype(np.float32), spread=lr.spread.astype(np.float32), building=lr.building,
        ndvi=ndvi if ndvi is not None else np.zeros(0, np.float32), has_ndvi=ndvi is not None,
        point_count=lr.point_count,
    )
    if cfg.run.save_rasters:
        write_raster(d / "chm.tif", lr.chm.astype(np.float32), grid)
        write_raster(d / "dtm.tif", lr.dtm.astype(np.float32), grid)
        if ndvi is not None:
            write_raster(d / "ndvi.tif", ndvi.astype(np.float32), grid)
    return TileRasters(grid, lr.chm.astype(np.float32), lr.spread.astype(np.float32), lr.building, ndvi, lr.point_count)


# -- stage B ------------------------------------------------------------------------------------

@dataclass
class Detection:
    crowns: gpd.GeoDataFrame
    mask: np.ndarray
    labels: np.ndarray
    n_tops: int


def detect(cfg: Config, tile: Tile, r: TileRasters) -> Detection:
    res = r.grid.res
    chm_s = smooth_chm(r.chm, cfg.treetops.smooth_sigma, res)
    mask = vegetation_mask(chm_s, r.spread, r.building, r.ndvi, res, cfg.mask)
    rows, cols = find_treetops(chm_s, mask, res, cfg.treetops)
    labels = segment_crowns(chm_s, mask, rows, cols, res, cfg.segment)
    crowns = crowns_to_polygons(labels, r.chm, r.grid, rows, cols, tile, cfg.polygons)
    return Detection(crowns=crowns, mask=mask, labels=labels, n_tops=int(rows.size))


def process_tile(cfg: Config, tile: Tile) -> dict:
    """Run both stages for one tile, skipping work that is already done with the same settings."""
    d = tile_dir(cfg, tile.id)
    done_file = d / "done.json"
    key = {"rasters": settings_hash(cfg, RASTER_SECTIONS), "detect": settings_hash(cfg, DETECT_SECTIONS)}
    if done_file.exists() and not cfg.run.overwrite:
        done = json.loads(done_file.read_text())
        if done.get("key") == key:
            return {**done, "skipped": True}

    r = prepare_rasters(cfg, tile, AhnSource(cfg))
    det = detect(cfg, tile, r)

    crowns_file = d / "crowns.gpkg"
    crowns_file.unlink(missing_ok=True)
    if len(det.crowns):
        det.crowns.to_file(crowns_file, layer="crowns", driver="GPKG")
    if cfg.run.save_rasters:
        write_raster(d / "mask.tif", det.mask.astype(np.uint8), r.grid)
        write_raster(d / "crowns.tif", det.labels.astype(np.int32), r.grid)

    summary = {
        "tile_id": tile.id,
        "key": key,
        "points": r.point_count,
        "has_ndvi": r.ndvi is not None,
        "tops": det.n_tops,
        "crowns": len(det.crowns),
    }
    done_file.write_text(json.dumps(summary, indent=2))
    return {**summary, "skipped": False}


def _safe_process(cfg: Config, tile: Tile) -> dict:
    try:
        return process_tile(cfg, tile)
    except Exception as e:  # keep going with other tiles; report at the end
        return {"tile_id": tile.id, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def run_tiles(cfg: Config, tiles: list[Tile], workers: int | None = None) -> list[dict]:
    """Process tiles in parallel. AHN sheets are downloaded first so workers don't race for them."""
    AhnSource(cfg).prefetch([t.buffered for t in tiles])
    workers = workers or cfg.run.workers
    results: list[dict] = []
    if workers <= 1:
        for t in tqdm(tiles, desc="tiles"):
            results.append(_safe_process(cfg, t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_safe_process, cfg, t) for t in tiles]
            for f in tqdm(as_completed(futures), total=len(futures), desc="tiles"):
                results.append(f.result())
    for r in results:
        if "error" in r:
            log.error("Tile %s failed: %s\n%s", r["tile_id"], r["error"], r.get("traceback", ""))
    return results


# -- merge --------------------------------------------------------------------------------------

def merge_tiles(cfg: Config, tiles: list[Tile], boundary: BaseGeometry, out_path: Path) -> gpd.GeoDataFrame:
    """Combine tile results into one GeoPackage with a 'crowns' and a 'treetops' layer.
    A crown is kept when its tree top lies inside the boundary."""
    parts = []
    for t in tiles:
        f = tile_dir(cfg, t.id) / "crowns.gpkg"
        if f.exists():
            parts.append(gpd.read_file(f, layer="crowns"))
    crowns = pd.concat(parts, ignore_index=True) if parts else empty_crowns()
    crowns = gpd.GeoDataFrame(crowns, geometry="geometry", crs=CRS)
    if len(crowns):
        tops = gpd.points_from_xy(crowns["top_x"], crowns["top_y"], crs=CRS)
        crowns = crowns.loc[tops.within(boundary)].sort_values(["tile_id", "top_y", "top_x"], ascending=[True, False, True])
    crowns = crowns.reset_index(drop=True)
    crowns.insert(0, "tree_id", np.arange(1, len(crowns) + 1))
    crowns = crowns[["tree_id", *COLUMNS, "geometry"]]

    treetops = gpd.GeoDataFrame(
        crowns.drop(columns="geometry"), geometry=gpd.points_from_xy(crowns["top_x"], crowns["top_y"]), crs=CRS
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    crowns.to_file(out_path, layer="crowns", driver="GPKG")
    treetops.to_file(out_path, layer="treetops", driver="GPKG")
    return crowns


def write_raster(path: Path, data: np.ndarray, grid: Grid) -> None:
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=1, dtype=data.dtype, crs=CRS,
                   transform=grid.transform, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
