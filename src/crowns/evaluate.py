"""Compare detected crowns with corrected reference crowns in sample plots, and tune parameters.

Only plots whose `status` is "done" are used. A crown is scored in a plot when its centroid lies
inside that plot. Detected and reference crowns are paired one-to-one by overlap
(IoU = intersection / union); a pair counts as a match when IoU >= evaluate.iou_threshold.
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
import shapely
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .config import CRS, Config
from .pipeline import DETECT_SECTIONS, detect, prepare_rasters
from .polygons import max_extent
from .reference import CROWN_LAYER, DRAFT_LAYER, PLOT_LAYER, reference_path
from .sources.ahn import AhnSource
from .tiling import Tile

log = logging.getLogger(__name__)

SIZE_CLASSES = [(0, 20, "<20 m2"), (20, 80, "20-80 m2"), (80, np.inf, ">=80 m2")]
SPLITS = ("test", "tune", "all")


# -- loading -----------------------------------------------------------------------------------

@dataclass
class Reference:
    plots: gpd.GeoDataFrame  # finished plots only
    crowns: gpd.GeoDataFrame  # corrected crowns
    draft: gpd.GeoDataFrame  # untouched draft crowns


def load_reference(cfg: Config, split: str) -> Reference:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    path = reference_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; create it with `crowns sample-plots`")
    plots = gpd.read_file(path, layer=PLOT_LAYER)
    done = plots["status"].fillna("").astype(str).str.strip().str.lower() == "done"
    if split != "all":
        done &= plots["split"] == split
    plots = plots.loc[done].reset_index(drop=True)
    if plots.empty:
        raise ValueError(f"No plots with status 'done' in the {split!r} split of {path}")
    crowns = _clean(gpd.read_file(path, layer=CROWN_LAYER))
    draft = _clean(gpd.read_file(path, layer=DRAFT_LAYER))
    return Reference(plots=plots, crowns=crowns, draft=draft)


def _clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop empty geometries and repair invalid ones (hand-drawn polygons can self-intersect)."""
    gdf = gdf.set_crs(CRS) if gdf.crs is None else gdf.to_crs(CRS)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = shapely.make_valid(gdf.geometry.values)
    return gdf.reset_index(drop=True)


def tiles_for_plots(tiles: list[Tile], plots: gpd.GeoDataFrame, margin: float) -> list[Tile]:
    """Tiles whose crowns can reach into the plots."""
    area = plots.geometry.buffer(margin).union_all()
    return [t for t in tiles if shapely.box(*t.core).intersects(area)]


# -- matching ------------------------------------------------------------------------------------

def optimal_matching(ia: np.ndarray, ib: np.ndarray, cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one matching over candidate pairs (ia[k], ib[k]) with the lowest total cost.
    The pairs form a sparse graph; each connected part is solved exactly (Hungarian algorithm).
    Returns the chosen positions into ia/ib/cost as index arrays a, b and their costs."""
    empty = np.empty(0, dtype=int)
    if ia.size == 0:
        return empty, empty, np.empty(0)
    ua, ra = np.unique(ia, return_inverse=True)
    ub, rb = np.unique(ib, return_inverse=True)
    n = len(ua) + len(ub)
    graph = coo_matrix((np.ones(ia.size), (ra, rb + len(ua))), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    pair_comp = comp[ra]
    order = np.argsort(pair_comp, kind="stable")
    out_a, out_b, out_c = [], [], []
    for idx in np.split(order, np.flatnonzero(np.diff(pair_comp[order])) + 1):
        la, lra = np.unique(ra[idx], return_inverse=True)
        lb, lrb = np.unique(rb[idx], return_inverse=True)
        m = np.full((len(la), len(lb)), 1e9)
        m[lra, lrb] = cost[idx]
        r, c = linear_sum_assignment(m)
        ok = m[r, c] < 1e9
        out_a.append(ua[la[r[ok]]])
        out_b.append(ub[lb[c[ok]]])
        out_c.append(m[r[ok], c[ok]])
    return np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_c)


def match_polygons(ref: gpd.GeoDataFrame, pred: gpd.GeoDataFrame, iou_threshold: float):
    """Pairs (ref index, pred index, IoU) maximising the number of matches, then the overlap."""
    empty = np.empty(0, dtype=int)
    if len(ref) == 0 or len(pred) == 0:
        return empty, empty, np.empty(0)
    ri, pi = pred.sindex.query(ref.geometry, predicate="intersects")
    a = ref.geometry.values[ri]
    b = pred.geometry.values[pi]
    inter = shapely.area(shapely.intersection(a, b))
    iou = inter / (shapely.area(a) + shapely.area(b) - inter)
    ok = iou >= iou_threshold
    ia, ib, cost = optimal_matching(ri[ok], pi[ok], 1 - iou[ok])
    return ia, ib, 1 - cost


# -- evaluation ----------------------------------------------------------------------------------

@dataclass
class EvalResult:
    metrics: dict
    reference: gpd.GeoDataFrame  # reference crowns with status and IoU
    predictions: gpd.GeoDataFrame  # detected crowns with status
    plots: gpd.GeoDataFrame  # per-plot metrics


def _assign_plots(crowns: gpd.GeoDataFrame, plots: gpd.GeoDataFrame) -> np.ndarray:
    """plot_id of the plot containing each crown's centroid, or None."""
    out = np.full(len(crowns), None, dtype=object)
    if len(crowns) == 0:
        return out
    ci, pj = plots.sindex.query(crowns.geometry.centroid, predicate="within")
    out[ci] = plots["plot_id"].to_numpy()[pj]
    return out


def _prf(tp_ref: int, n_ref: int, tp_pred: int, n_pred: int) -> dict:
    p = tp_pred / n_pred if n_pred else None
    r = tp_ref / n_ref if n_ref else None
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else (0.0 if n_ref and n_pred else None)
    return {"precision": _round(p), "recall": _round(r), "f1": _round(f1)}


def _round(v):
    return None if v is None else round(float(v), 4)


def _contained(container: gpd.GeoDataFrame, items: gpd.GeoDataFrame, items_mask: np.ndarray, share: float = 0.5):
    """For each container polygon, how many (masked) items have >= `share` of their area inside it."""
    counts = np.zeros(len(container), dtype=int)
    if len(container) == 0 or not items_mask.any():
        return counts
    sel = np.flatnonzero(items_mask)
    ii, cj = container.sindex.query(items.geometry.values[sel], predicate="intersects")
    a = items.geometry.values[sel][ii]
    inside = shapely.area(shapely.intersection(a, container.geometry.values[cj])) >= share * shapely.area(a)
    np.add.at(counts, cj[inside], 1)
    return counts


def edit_summary(ref: Reference) -> dict:
    """How much of the draft was changed inside the finished plots. Many unchanged crowns can mean
    the draft was good, or that it was accepted without a close look."""
    area = ref.plots.union_all()
    crowns = ref.crowns.loc[ref.crowns.intersects(area)]
    draft = ref.draft.loc[ref.draft.intersects(area)]
    draft = draft.set_index(draft["draft_id"].astype(int))
    has_id = crowns["draft_id"].notna() & crowns["draft_id"].isin(draft.index)
    unchanged = np.zeros(len(crowns), dtype=bool)
    if has_id.any():
        ids = crowns.loc[has_id, "draft_id"].astype(int)
        dup = ids.duplicated(keep=False)  # a split crown keeps its id in both parts
        same = shapely.equals_exact(crowns.loc[has_id].geometry.values, draft.loc[ids].geometry.values, 0.01)
        unchanged[np.flatnonzero(has_id.to_numpy())] = same & ~dup.to_numpy()
    kept_ids = set(crowns["draft_id"].dropna().astype(int))
    return {
        "reference_crowns": int(len(crowns)),
        "unchanged": int(unchanged.sum()),
        "reshaped": int((has_id.to_numpy() & ~unchanged).sum()),
        "added": int((~has_id).sum()),
        "deleted": int(sum(1 for i in draft.index if int(i) not in kept_ids)),
        "unchanged_share": _round(unchanged.mean()) if len(crowns) else None,
    }


def evaluate(cfg: Config, ref: Reference, pred: gpd.GeoDataFrame) -> EvalResult:
    plots = ref.plots
    area = plots.union_all()
    rc = ref.crowns.loc[ref.crowns.intersects(area)].reset_index(drop=True)
    pc = pred.loc[pred.intersects(area)].reset_index(drop=True)
    rc["plot_id"] = _assign_plots(rc, plots)
    pc["plot_id"] = _assign_plots(pc, plots)
    r_in = rc["plot_id"].notna().to_numpy()
    p_in = pc["plot_id"].notna().to_numpy()

    ia, ib, iou = match_polygons(rc, pc, cfg.evaluate.iou_threshold)
    r_match = np.zeros(len(rc), bool)
    p_match = np.zeros(len(pc), bool)
    r_match[ia] = True
    p_match[ib] = True
    rc["iou"] = np.nan
    rc.loc[ia, "iou"] = np.round(iou, 3)
    rc["status"] = np.where(~r_in, "outside", np.where(r_match, "matched", "missed"))
    pc["status"] = np.where(~p_in, "outside", np.where(p_match, "matched", "false"))

    # Merged: one detected crown covering most of two or more reference crowns. Split: the reverse.
    merges = _contained(pc, rc, r_in)
    splits = _contained(rc, pc, np.ones(len(pc), bool))
    pc["n_ref_inside"] = merges
    rc["n_pred_inside"] = splits

    n_ref, n_pred = int(r_in.sum()), int(p_in.sum())
    tp_ref, tp_pred = int((r_in & r_match).sum()), int((p_in & p_match).sum())
    matched_iou = rc.loc[r_in & r_match, "iou"]

    pred_canopy = pc.geometry.union_all().intersection(area) if len(pc) else shapely.Polygon()
    ref_canopy = rc.geometry.union_all().intersection(area) if len(rc) else shapely.Polygon()
    overlap = pred_canopy.intersection(ref_canopy).area

    metrics: dict = {
        "plots": int(len(plots)),
        "reference_crowns": n_ref,
        "detected_crowns": n_pred,
        "matched": tp_ref,
        **_prf(tp_ref, n_ref, tp_pred, n_pred),
        "mean_iou_matched": _round(matched_iou.mean()) if len(matched_iou) else None,
        "merged_detections": int((p_in & (merges >= 2)).sum()),
        "split_references": int((r_in & (splits >= 2)).sum()),
        "canopy_area_precision": _round(overlap / pred_canopy.area) if pred_canopy.area else None,
        "canopy_area_recall": _round(overlap / ref_canopy.area) if ref_canopy.area else None,
    }

    # Size accuracy of matched crowns: detected minus reference. A negative bias means the
    # detected crowns are too small.
    rc["area_m2"] = np.round(shapely.area(rc.geometry.values), 2)
    rc["diam_max"] = np.round([max_extent(g) for g in rc.geometry.values], 2)
    rc["pred_area_m2"] = np.nan
    rc["pred_diam_max"] = np.nan
    pairs = r_in[ia]
    if pairs.any():
        ref_rows, pred_rows = ia[pairs], ib[pairs]
        ref_a = rc["area_m2"].to_numpy()[ref_rows]
        pred_a = shapely.area(pc.geometry.values[pred_rows])
        ref_d, pred_d = 2 * np.sqrt(ref_a / np.pi), 2 * np.sqrt(pred_a / np.pi)
        ref_x = rc["diam_max"].to_numpy()[ref_rows]
        pred_x = np.array([max_extent(g) for g in pc.geometry.values[pred_rows]])
        rel = (pred_a - ref_a) / ref_a
        metrics.update({
            "area_bias_pct": _round(100 * rel.mean()),
            "area_mae_pct": _round(100 * np.abs(rel).mean()),
            "diam_eq_bias_m": _round((pred_d - ref_d).mean()),
            "diam_eq_rmse_m": _round(np.sqrt(((pred_d - ref_d) ** 2).mean())),
            "diam_max_bias_m": _round((pred_x - ref_x).mean()),
            "diam_max_rmse_m": _round(np.sqrt(((pred_x - ref_x) ** 2).mean())),
        })
        rc.loc[ref_rows, "pred_area_m2"] = np.round(pred_a, 2)
        rc.loc[ref_rows, "pred_diam_max"] = np.round(pred_x, 2)

    # Recall by crown size, and scores by canopy-cover class.
    r_area = rc.geometry.area.to_numpy()
    metrics["recall_by_size"] = {
        name: _round((r_in & r_match & (r_area >= lo) & (r_area < hi)).sum() / n)
        if (n := int((r_in & (r_area >= lo) & (r_area < hi)).sum())) else None
        for lo, hi, name in SIZE_CLASSES
    }
    cover_of = plots.set_index("plot_id")["cover_class"]
    by_cover = {}
    for cls in sorted(plots["cover_class"].unique()):
        ids = set(cover_of.index[cover_of == cls])
        rm, pm = rc["plot_id"].isin(ids).to_numpy(), pc["plot_id"].isin(ids).to_numpy()
        by_cover[cls] = _prf(int((rm & r_match).sum()), int(rm.sum()), int((pm & p_match).sum()), int(pm.sum()))
    metrics["by_cover_class"] = by_cover
    metrics["edits"] = edit_summary(ref)

    # Per-plot table, to find the plots where the model does worst.
    rows = []
    for p in plots.itertuples():
        rm, pm = (rc["plot_id"] == p.plot_id).to_numpy(), (pc["plot_id"] == p.plot_id).to_numpy()
        rows.append({"plot_id": p.plot_id, "n_ref": int(rm.sum()), "n_pred": int(pm.sum()),
                     **_prf(int((rm & r_match).sum()), int(rm.sum()), int((pm & p_match).sum()), int(pm.sum()))})
    plot_table = plots[["plot_id", "split", "cover_class", "geometry"]].merge(pd.DataFrame(rows), on="plot_id")
    return EvalResult(metrics=metrics, reference=rc, predictions=pc, plots=plot_table)


def write_evaluation(result: EvalResult, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / f"{name}.gpkg"
    gpkg.unlink(missing_ok=True)
    result.plots.to_file(gpkg, layer="plots", driver="GPKG")
    result.reference.to_file(gpkg, layer="reference", driver="GPKG")
    result.predictions.to_file(gpkg, layer="predictions", driver="GPKG")
    (out_dir / f"{name}.json").write_text(json.dumps(result.metrics, indent=2))
    return gpkg


# -- tuning --------------------------------------------------------------------------------------

def tune(cfg: Config, tiles: list[Tile]) -> pd.DataFrame:
    """Try every combination in tune.grid on the 'tune' plots; best F1 first (then mean IoU).
    Only detection settings can be tuned; rasters are built once per tile and reused."""
    grid = cfg.tune.grid
    if not grid:
        raise ValueError("Set tune.grid, e.g. {'treetops.window_b': [0.06, 0.08, 0.1]}")
    bad = [k for k in grid if k.split(".")[0] not in DETECT_SECTIONS]
    if bad:
        raise ValueError(f"Only {', '.join(DETECT_SECTIONS)} settings can be tuned, not: {', '.join(bad)}")
    ref = load_reference(cfg, "tune")
    needed = tiles_for_plots(tiles, ref.plots, cfg.tiling.buffer)

    ahn = AhnSource(cfg)
    ahn.prefetch([t.buffered for t in needed])
    rasters = {t.id: prepare_rasters(cfg, t, ahn) for t in needed}

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    rows = []
    for i, values in enumerate(combos, start=1):
        overrides = dict(zip(keys, values))
        c = cfg.with_overrides(overrides)
        pred = pd.concat([detect(c, t, rasters[t.id]).crowns for t in needed], ignore_index=True)
        pred = gpd.GeoDataFrame(pred, geometry="geometry", crs=CRS)
        m = evaluate(c, ref, pred).metrics
        rows.append({**overrides, **{k: v for k, v in m.items() if not isinstance(v, dict)}})
        log.info("[%d/%d] %s -> F1 %s, IoU %s", i, len(combos), overrides, m["f1"], m["mean_iou_matched"])
    df = pd.DataFrame(rows)
    return df.sort_values(["f1", "mean_iou_matched"], ascending=False, na_position="last").reset_index(drop=True)
