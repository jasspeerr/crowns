"""Compare detected trees with a tree register (points), and tune parameters on a few tiles.

The register only lists public trees, so a detection without a register tree is only counted as
a false positive inside `zones`: polygons where every tree should be in the register (streets,
public parks). Recall (share of register trees found) is measured everywhere.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import CRS, Config
from .pipeline import DETECT_SECTIONS, detect, prepare_rasters, tile_dir
from .sources.ahn import AhnSource
from .tiling import Grid, Tile

log = logging.getLogger(__name__)


def load_register(cfg: Config) -> gpd.GeoDataFrame:
    e = cfg.evaluate
    if not e.register_file:
        raise ValueError("Set evaluate.register_file")
    path = cfg.path(e.register_file)
    gdf = gpd.read_file(path, layer=e.register_layer) if e.register_layer else gpd.read_file(path)
    gdf = gdf.set_crs(CRS) if gdf.crs is None else gdf.to_crs(CRS)
    if not (gdf.geom_type == "Point").all():
        gdf["geometry"] = gdf.geometry.representative_point()
    return gdf.reset_index(drop=True)


def load_zones(cfg: Config) -> BaseGeometry | None:
    if not cfg.evaluate.zones_file:
        return None
    gdf = gpd.read_file(cfg.path(cfg.evaluate.zones_file))
    gdf = gdf.set_crs(CRS) if gdf.crs is None else gdf.to_crs(CRS)
    return gdf.union_all()


def match_points(a: np.ndarray, b: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal one-to-one matching of point sets a and b (Nx2, Mx2) within max_dist.

    The candidate pairs form a sparse graph; each connected part is solved exactly with the
    Hungarian algorithm, which keeps it fast for a whole municipality.
    Returns indices into a, indices into b, and pair distances."""
    empty = np.empty(0, dtype=int)
    if len(a) == 0 or len(b) == 0:
        return empty, empty, np.empty(0)
    near = cKDTree(b).query_ball_point(a, r=max_dist)
    counts = np.fromiter((len(n) for n in near), dtype=int, count=len(a))
    if counts.sum() == 0:
        return empty, empty, np.empty(0)
    pairs_a = np.repeat(np.arange(len(a)), counts)
    pairs_b = np.fromiter(itertools.chain.from_iterable(near), dtype=int, count=int(counts.sum()))
    dist = np.sqrt(((a[pairs_a] - b[pairs_b]) ** 2).sum(1))

    na = len(a)
    graph = coo_matrix((np.ones(len(pairs_a)), (pairs_a, pairs_b + na)), shape=(na + len(b), na + len(b)))
    _, comp = connected_components(graph, directed=False)

    out_a, out_b, out_d = [], [], []
    comp_of_pair = comp[pairs_a]
    order = np.argsort(comp_of_pair, kind="stable")
    bounds = np.flatnonzero(np.diff(comp_of_pair[order])) + 1
    for idx in np.split(order, bounds):
        ia, ib, d = pairs_a[idx], pairs_b[idx], dist[idx]
        ua, ra = np.unique(ia, return_inverse=True)
        ub, rb = np.unique(ib, return_inverse=True)
        cost = np.full((len(ua), len(ub)), 1e9)
        cost[ra, rb] = d
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < 1e9
        out_a.append(ua[r[ok]])
        out_b.append(ub[c[ok]])
        out_d.append(cost[r[ok], c[ok]])
    return np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_d)


def chm_max_near(cfg: Config, tiles: list[Tile], x: np.ndarray, y: np.ndarray, radius: float) -> np.ndarray:
    """Highest CHM value within `radius` of each point, from the cached tile rasters. NaN if unknown."""
    out = np.full(len(x), np.nan)
    for t in tiles:
        f = tile_dir(cfg, t.id) / "rasters.npz"
        sel = np.flatnonzero(t.owns(x, y))
        if sel.size == 0 or not f.exists():
            continue
        with np.load(f) as z:
            chm = z["chm"]
        grid = Grid.from_bounds(t.buffered, cfg.lidar.resolution)
        r = int(np.ceil(radius / grid.res))
        row, col, inside = grid.cell_index(x[sel], y[sel])
        for i, rr, cc, ok in zip(sel, row, col, inside):
            if ok:
                out[i] = chm[max(rr - r, 0): rr + r + 1, max(cc - r, 0): cc + r + 1].max()
    return out


@dataclass
class EvalResult:
    metrics: dict
    register: gpd.GeoDataFrame  # register points with a 'status' column
    treetops: gpd.GeoDataFrame  # detected tops with a 'status' column


def evaluate(cfg: Config, crowns: gpd.GeoDataFrame, register: gpd.GeoDataFrame, tiles: list[Tile],
             area: BaseGeometry, zones: BaseGeometry | None) -> EvalResult:
    md = cfg.evaluate.match_distance
    reg = register.loc[register.within(area)].reset_index(drop=True)
    crowns = crowns.reset_index(drop=True)
    tops = gpd.GeoDataFrame(crowns.drop(columns="geometry"),
                            geometry=gpd.points_from_xy(crowns["top_x"], crowns["top_y"]), crs=CRS)
    tops = tops.loc[tops.within(area)]
    crowns = crowns.loc[tops.index].reset_index(drop=True)
    tops = tops.reset_index(drop=True)

    a = np.c_[reg.geometry.x, reg.geometry.y]
    b = np.c_[tops.geometry.x, tops.geometry.y]
    ia, ib, dist = match_points(a, b, md)

    reg["status"] = "missed"
    reg.loc[ia, "status"] = "matched"
    reg["match_dist"] = np.nan
    reg.loc[ia, "match_dist"] = dist

    # Register trees that are missed and have no canopy of tree height nearby: just planted,
    # removed, or too small to detect. Reported separately, not counted as misses.
    missed = np.flatnonzero(reg["status"] == "missed")
    if missed.size:
        h = chm_max_near(cfg, tiles, a[missed, 0], a[missed, 1], md)
        small = missed[np.nan_to_num(h, nan=0.0) < cfg.treetops.min_height]
        reg.loc[small, "status"] = "small_or_absent"

    # Register points inside a detected crown, regardless of the one-to-one matching.
    in_crown = gpd.sjoin(reg[["geometry"]], crowns[["geometry"]], predicate="within", how="inner")
    reg["in_crown"] = reg.index.isin(in_crown.index)
    per_crown = in_crown.groupby("index_right").size()

    tops["status"] = "unmatched"
    tops.loc[ib, "status"] = "matched"

    # Extra tops close to a matched register tree suggest one tree was split into several crowns.
    if len(ia) and len(tops):
        near = cKDTree(a[ia]).query_ball_point(b, md)
        extra = np.array([len(n) > 0 for n in near]) & (tops["status"] == "unmatched").to_numpy()
    else:
        extra = np.zeros(len(tops), dtype=bool)

    countable = reg["status"] != "small_or_absent"
    n_reg = int(countable.sum())
    n_match = int((reg["status"] == "matched").sum())
    metrics: dict = {
        "register_trees": int(len(reg)),
        "register_small_or_absent": int((~countable).sum()),
        "detected_trees": int(len(tops)),
        "matched": n_match,
        "recall": _ratio(n_match, n_reg),
        "in_crown_rate": _ratio(int((reg["in_crown"] & countable).sum()), n_reg),
        "merged_crowns": int((per_crown >= 2).sum()),
        "trees_in_merged_crowns": int(per_crown[per_crown >= 2].sum()),
        "split_extra_tops": int(extra.sum()),
        "mean_match_distance": float(np.mean(dist)) if len(dist) else None,
    }

    if zones is not None:
        reg_z = reg.within(zones) & countable
        tops_z = tops.within(zones)
        tp_z = int((reg_z & (reg["status"] == "matched")).sum())
        p = _ratio(int((tops_z & (tops["status"] == "matched")).sum()), int(tops_z.sum()))
        r = _ratio(tp_z, int(reg_z.sum()))
        tops.loc[~tops_z & (tops["status"] == "unmatched"), "status"] = "outside_zones"
        metrics.update({
            "zone_register_trees": int(reg_z.sum()),
            "zone_detected_trees": int(tops_z.sum()),
            "zone_precision": p,
            "zone_recall": r,
            "zone_f1": _ratio(2 * p * r, p + r) if p is not None and r is not None else None,
        })
    else:
        tops.loc[tops["status"] == "unmatched", "status"] = "unknown"
    return EvalResult(metrics=metrics, register=reg, treetops=tops)


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def evaluated_area(tiles: list[Tile], boundary: BaseGeometry) -> BaseGeometry:
    return unary_union([box(*t.core) for t in tiles]).intersection(boundary)


def write_evaluation(result: EvalResult, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / f"{name}.gpkg"
    gpkg.unlink(missing_ok=True)
    result.register.to_file(gpkg, layer="register", driver="GPKG")
    if len(result.treetops):
        result.treetops.to_file(gpkg, layer="treetops", driver="GPKG")
    (out_dir / f"{name}.json").write_text(json.dumps(result.metrics, indent=2))
    return gpkg


def tune(cfg: Config, tiles: list[Tile], boundary: BaseGeometry) -> pd.DataFrame:
    """Try every combination in cfg.tune.grid on the given tiles and rank by F1 in the zones.

    Only detection settings can be tuned; rasters are built once per tile and reused."""
    grid = cfg.tune.grid
    if not grid:
        raise ValueError("Set tune.grid, e.g. {'treetops.window_b': [0.06, 0.08, 0.1]}")
    bad = [k for k in grid if k.split(".")[0] not in DETECT_SECTIONS]
    if bad:
        raise ValueError(f"Only {', '.join(DETECT_SECTIONS)} settings can be tuned, not: {', '.join(bad)}")
    zones = load_zones(cfg)
    if zones is None:
        raise ValueError("Tuning needs evaluate.zones_file: without it precision is unknown and "
                         "over-splitting would always win")
    register = load_register(cfg)
    area = evaluated_area(tiles, boundary)

    ahn = AhnSource(cfg)
    ahn.prefetch([t.buffered for t in tiles])
    rasters = {t.id: prepare_rasters(cfg, t, ahn) for t in tiles}

    keys = list(grid)
    rows = []
    combos = list(itertools.product(*(grid[k] for k in keys)))
    for i, values in enumerate(combos, start=1):
        overrides = dict(zip(keys, values))
        c = cfg.with_overrides(overrides)
        crowns = pd.concat([detect(c, t, rasters[t.id]).crowns for t in tiles], ignore_index=True)
        crowns = gpd.GeoDataFrame(crowns, geometry="geometry", crs=CRS)
        m = evaluate(c, crowns, register, tiles, area, zones).metrics
        rows.append({**overrides, **m})
        log.info("[%d/%d] %s -> F1 %s", i, len(combos), overrides, m.get("zone_f1"))
    df = pd.DataFrame(rows)
    return df.sort_values("zone_f1", ascending=False, na_position="last").reset_index(drop=True)
