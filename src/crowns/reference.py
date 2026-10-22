"""Evaluation data from corrected draft results.

`sample_plots` picks square plots from processed tiles and writes a GeoPackage with:
  plots         one square per plot; set `status` to "done" once every crown in it is correct
  crowns        draft crowns around the plots; edit these (delete, add, reshape, split, merge)
  crowns_draft  an untouched copy, used to report how much was corrected

plus CHM and CIR rasters per plot under `<reference dir>/rasters/` as background for editing.

A crown belongs to a plot when its centroid lies inside the plot. Crowns crossing the plot edge are
included too, so they can be corrected, but only crowns centred inside a plot are scored.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .config import CRS, Config
from .pipeline import DETECT_SECTIONS, cir_path, settings_hash, tile_dir, write_raster
from .polygons import empty_crowns
from .tiling import Grid, Tile

log = logging.getLogger(__name__)

PLOT_LAYER = "plots"
CROWN_LAYER = "crowns"
DRAFT_LAYER = "crowns_draft"


def reference_path(cfg: Config) -> Path:
    return cfg.path(cfg.reference.file)  # type: ignore[return-value]


def processed_tiles(cfg: Config, tiles: list[Tile]) -> list[Tile]:
    return [t for t in tiles if (tile_dir(cfg, t.id) / "done.json").exists()]


def load_tile_crowns(cfg: Config, tiles: list[Tile]) -> gpd.GeoDataFrame:
    parts = [gpd.read_file(f, layer="crowns") for t in tiles if (f := tile_dir(cfg, t.id) / "crowns.gpkg").exists()]
    if not parts:
        return empty_crowns()
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS)


def candidate_plots(tiles: list[Tile], boundary: BaseGeometry, size: float) -> gpd.GeoDataFrame:
    """All plot squares on a grid inside the tile cores that lie completely inside the boundary."""
    rows = []
    for t in tiles:
        xmin, ymin, xmax, ymax = t.core
        for x in np.arange(xmin, xmax - size + 1e-6, size):
            for y in np.arange(ymin, ymax - size + 1e-6, size):
                rows.append((t.id, box(x, y, x + size, y + size)))
    if not rows:
        return gpd.GeoDataFrame({"tile_id": []}, geometry=[], crs=CRS)
    gdf = gpd.GeoDataFrame({"tile_id": [r[0] for r in rows]}, geometry=[r[1] for r in rows], crs=CRS)
    return gdf.loc[gdf.within(boundary)].reset_index(drop=True)


def canopy_cover(plots: gpd.GeoDataFrame, crowns: gpd.GeoDataFrame) -> np.ndarray:
    """Share of each plot covered by crowns (crowns hardly overlap, so areas are summed)."""
    cover = np.zeros(len(plots))
    if len(crowns) == 0 or len(plots) == 0:
        return cover
    pi, ci = crowns.sindex.query(plots.geometry, predicate="intersects")
    inter = plots.geometry.values[pi].intersection(crowns.geometry.values[ci]).area
    np.add.at(cover, pi, inter)
    return np.clip(cover / plots.geometry.area.to_numpy(), 0, 1)


def sample_plots(cfg: Config, tiles: list[Tile], boundary: BaseGeometry, crowns: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Random plots, spread evenly over canopy-cover classes, split into tune and test plots."""
    r = cfg.reference
    rng = np.random.default_rng(r.seed)
    cand = candidate_plots(tiles, boundary, r.plot_size)
    if cand.empty:
        raise ValueError("No room for plots: are the tiles processed and inside the boundary?")
    cand["cover"] = canopy_cover(cand, crowns)
    edges = list(r.cover_classes) + [np.inf]
    n_classes = len(edges) - 1
    per_class = [r.n_plots // n_classes + (1 if i < r.n_plots % n_classes else 0) for i in range(n_classes)]

    cls = np.full(len(cand), -1)
    for i in range(n_classes):
        cls[(cand["cover"] >= edges[i]).to_numpy() & (cand["cover"] < edges[i + 1]).to_numpy()] = i
    pools = [list(rng.permutation(np.flatnonzero(cls == i))) for i in range(n_classes)]

    # Take the planned number per class; when a class runs short, fill up from the classes that
    # still have plots left, so the total is reached whenever possible.
    picked: list[list[int]] = [[pools[i].pop() for _ in range(min(per_class[i], len(pools[i])))]
                               for i in range(n_classes)]
    short = r.n_plots - sum(map(len, picked))
    for i in range(n_classes):
        if len(picked[i]) < per_class[i]:
            log.warning("Only %d plots with canopy cover %s", len(picked[i]), _class_name(edges, i))
    while short > 0 and any(pools):
        for i in sorted(range(n_classes), key=lambda k: len(picked[k])):
            if pools[i] and short > 0:
                picked[i].append(pools[i].pop())
                short -= 1
    if short > 0:
        log.warning("Only %d plots available in total", r.n_plots - short)

    chosen = []
    for i in range(n_classes):
        take = len(picked[i])
        n_tune = int(round(take * r.tune_fraction))
        split = np.array(["tune"] * n_tune + ["test"] * (take - n_tune))
        rng.shuffle(split)
        sub = cand.iloc[picked[i]].copy()
        sub["cover_class"] = _class_name(edges, i)
        sub["split"] = split
        chosen.append(sub)

    plots = pd.concat(chosen).sort_values(["cover_class", "tile_id"]).reset_index(drop=True)
    plots.insert(0, "plot_id", [f"P{i + 1:03d}" for i in range(len(plots))])
    plots["cover"] = plots["cover"].round(3)
    plots["status"] = "todo"
    plots["notes"] = ""
    plots["draft_settings"] = settings_hash(cfg, DETECT_SECTIONS)
    return gpd.GeoDataFrame(plots, geometry="geometry", crs=CRS)


def _class_name(edges: list[float], i: int) -> str:
    return f"{100 * edges[i]:.0f}-{100 * min(edges[i + 1], 1):.0f}%"


def export_reference(cfg: Config, plots: gpd.GeoDataFrame, crowns: gpd.GeoDataFrame, tiles: list[Tile],
                     force: bool = False) -> Path:
    path = reference_path(cfg)
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists. It may hold your corrections; use --force to "
                                  "replace it (the old file is kept as a backup).")
        backup = path.with_name(f"{path.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
        shutil.move(path, backup)
        log.warning("Moved the existing reference file to %s", backup)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Draft crowns that touch any plot.
    hit = np.unique(crowns.sindex.query(plots.geometry, predicate="intersects")[1]) if len(crowns) else []
    draft = crowns.iloc[hit].reset_index(drop=True)
    draft = gpd.GeoDataFrame(
        {"draft_id": np.arange(1, len(draft) + 1), "h_max": draft.get("h_max", pd.Series(dtype=float))},
        geometry=draft.geometry.values, crs=CRS,
    )
    plots.to_file(path, layer=PLOT_LAYER, driver="GPKG")
    draft.to_file(path, layer=CROWN_LAYER, driver="GPKG")
    draft.to_file(path, layer=DRAFT_LAYER, driver="GPKG")

    by_id = {t.id: t for t in tiles}
    raster_dir = path.parent / "rasters"
    raster_dir.mkdir(exist_ok=True)
    for p in plots.itertuples():
        _export_plot_rasters(cfg, by_id[p.tile_id], p.plot_id, p.geometry.bounds, raster_dir)
    return path


def _export_plot_rasters(cfg: Config, tile: Tile, plot_id: str, bounds, out_dir: Path) -> None:
    m = cfg.reference.raster_margin
    b = (bounds[0] - m, bounds[1] - m, bounds[2] + m, bounds[3] + m)
    d = tile_dir(cfg, tile.id)
    with np.load(d / "rasters.npz") as z:
        chm = z["chm"]
    grid = Grid.from_bounds(tile.buffered, cfg.lidar.resolution)
    c0 = int(round((b[0] - grid.xmin) / grid.res))
    r0 = int(round((grid.ymax - b[3]) / grid.res))
    sub = Grid(xmin=grid.xmin + c0 * grid.res, ymax=grid.ymax - r0 * grid.res, res=grid.res,
               width=int(round((b[2] - b[0]) / grid.res)), height=int(round((b[3] - b[1]) / grid.res)))
    write_raster(out_dir / f"{plot_id}_chm.tif", chm[r0:r0 + sub.height, c0:c0 + sub.width], sub)

    cir = cir_path(cfg, tile.id)
    if cir.exists():
        with rasterio.open(cir) as src:
            window = from_bounds(*b, transform=src.transform)
            data = src.read(window=window, boundless=True, fill_value=0)
            profile = src.profile | dict(width=data.shape[2], height=data.shape[1],
                                         transform=src.window_transform(window), tiled=False)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        with rasterio.open(out_dir / f"{plot_id}_cir.tif", "w", **profile) as dst:
            dst.write(data)
