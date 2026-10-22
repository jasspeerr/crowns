"""Crown raster to polygons with attributes."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import Polygon, shape

from .config import CRS, PolygonConfig
from .tiling import Grid, Tile

COLUMNS = ["tile_id", "top_x", "top_y", "h_max", "h_mean", "area_m2", "diam_eq", "diam_max"]


def empty_crowns() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({c: [] for c in COLUMNS}, geometry=[], crs=CRS)


def crowns_to_polygons(
    labels: np.ndarray,
    chm: np.ndarray,
    grid: Grid,
    top_rows: np.ndarray,
    top_cols: np.ndarray,
    tile: Tile,
    cfg: PolygonConfig,
) -> gpd.GeoDataFrame:
    """One polygon per crown whose tree top lies in the tile core."""
    n = top_rows.size
    if n == 0 or not labels.any():
        return empty_crowns()

    top_x, top_y = grid.xy(top_rows, top_cols)
    present = np.zeros(n + 1, dtype=bool)
    present[np.unique(labels)] = True
    # Only crowns owned by this tile; the buffer is there so these are complete.
    wanted = tile.owns(top_x, top_y) & present[1:]
    if not wanted.any():
        return empty_crowns()

    lab = np.where(np.concatenate([[False], wanted])[labels], labels, 0).astype(np.int32)
    geoms: dict[int, Polygon] = {}
    for geom, value in shapes(lab, mask=lab > 0, transform=grid.transform):
        poly = shape(geom)
        k = int(value)
        # Each crown is one connected piece, but keep the largest if pixels touch only at a corner.
        if k not in geoms or poly.area > geoms[k].area:
            geoms[k] = poly

    keys = np.array(sorted(geoms))
    polys = np.array([Polygon(geoms[k].exterior) for k in keys], dtype=object)  # drop holes
    if cfg.smooth > 0:
        polys = shapely.buffer(shapely.buffer(polys, cfg.smooth, join_style="round"), -cfg.smooth, join_style="round")
    if cfg.simplify > 0:
        polys = shapely.simplify(polys, cfg.simplify, preserve_topology=True)
    polys = np.array([_largest_polygon(p) for p in polys], dtype=object)

    h_max = np.asarray(ndimage.maximum(chm, lab, keys))
    h_mean = np.asarray(ndimage.mean(chm, lab, keys))
    area = shapely.area(polys)
    gdf = gpd.GeoDataFrame(
        {
            "tile_id": tile.id,
            "top_x": np.round(top_x[keys - 1], 2),
            "top_y": np.round(top_y[keys - 1], 2),
            "h_max": np.round(h_max, 2),
            "h_mean": np.round(h_mean, 2),
            "area_m2": np.round(area, 2),
            "diam_eq": np.round(2 * np.sqrt(area / np.pi), 2),
            "diam_max": np.round([max_extent(p) for p in polys], 2),
        },
        geometry=list(polys),
        crs=CRS,
    )
    keep = (~gdf.geometry.is_empty) & (gdf["area_m2"] >= cfg.min_area)
    return gdf.loc[keep].reset_index(drop=True)


def _largest_polygon(geom) -> Polygon:
    if geom is None or geom.is_empty:
        return Polygon()
    if geom.geom_type == "Polygon":
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return max(parts, key=lambda g: g.area) if parts else Polygon()


def max_extent(poly: Polygon) -> float:
    """Longest distance across the crown (between two points of its convex hull)."""
    if poly.is_empty:
        return 0.0
    xy = np.asarray(poly.convex_hull.exterior.coords) if poly.convex_hull.geom_type == "Polygon" else np.zeros((1, 2))
    d = xy[:, None, :] - xy[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())
