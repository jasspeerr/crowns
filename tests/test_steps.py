import numpy as np
import pytest
from shapely.geometry import Point, box

from crowns.config import (MaskConfig, PolygonConfig, SegmentConfig, TreetopConfig, config_from_dict)
from crowns.evaluate import match_points
from crowns.lidar import fill_gaps, grid_count, grid_max, grid_min
from crowns.mask import vegetation_mask
from crowns.polygons import crowns_to_polygons
from crowns.segment import segment_crowns
from crowns.tiling import Grid, Tile, make_tiles
from crowns.treetops import find_treetops, smooth_chm

RES = 0.5


def paraboloid_chm(trees, size=100.0, res=RES):
    """CHM with paraboloid crowns: (x, y, height, radius) in metres from the top-left corner."""
    n = int(size / res)
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    x, y = (cols + 0.5) * res, (rows + 0.5) * res
    chm = np.zeros((n, n))
    for tx, ty, h, r in trees:
        d = np.hypot(x - tx, y - ty)
        chm = np.maximum(chm, np.where(d <= r, h - 0.6 * h * (d / r) ** 2, 0))
    return chm


TREES = [(20, 20, 15, 4), (60, 30, 22, 6), (30, 70, 9, 3), (70, 70, 16, 4.5), (77, 70, 13, 4)]


# -- tiling ---------------------------------------------------------------------------------

def test_tiles_cover_boundary_and_own_points_once():
    tiles = make_tiles(box(10, 10, 950, 420), size=500, buffer=25)
    assert sorted(t.id for t in tiles) == ["0_0", "500_0"]
    x = np.array([500.0, 499.99])
    y = np.array([100.0, 100.0])
    owners = [t.owns(x, y) for t in tiles]
    assert (np.sum(owners, axis=0) == 1).all()  # each point has exactly one owner


def test_grid_roundtrip():
    g = Grid.from_bounds((1000, 2000, 1010, 2004), 0.5)
    assert g.shape == (8, 20)
    r, c, inside = g.cell_index(np.array([1000.1, 1009.9]), np.array([2003.9, 2000.1]))
    assert inside.all() and r.tolist() == [0, 7] and c.tolist() == [0, 19]
    x, y = g.xy(r, c)
    assert np.allclose(x, [1000.25, 1009.75]) and np.allclose(y, [2003.75, 2000.25])


# -- rasterising ----------------------------------------------------------------------------

def test_grid_reducers():
    g = Grid.from_bounds((0, 0, 2, 2), 1.0)
    x = np.array([0.5, 0.6, 1.5])
    y = np.array([1.5, 1.5, 0.5])
    v = np.array([1.0, 3.0, 2.0])
    mx, mn, cnt = grid_max(g, x, y, v), grid_min(g, x, y, v), grid_count(g, x, y)
    assert mx[0, 0] == 3 and mn[0, 0] == 1 and mx[1, 1] == 2
    assert np.isnan(mx[0, 1]) and cnt.sum() == 3


def test_fill_gaps_keeps_zero_heights():
    # Terrain around 0 m NAP: a real 0.0 must not be mistaken for "not filled".
    r = np.full((20, 20), np.nan)
    r[:, :10] = 0.0
    r[:, 10:12] = 0.2
    filled = fill_gaps(r, max_search_cells=3)
    assert np.isfinite(filled).all()
    assert np.all(filled[:, :10] == 0.0)
    assert filled[:, -1].max() <= 0.2 + 1e-6


# -- detection -------------------------------------------------------------------------------

def _detect(chm):
    tcfg = TreetopConfig()
    chm_s = smooth_chm(chm, tcfg.smooth_sigma, RES)
    mask = vegetation_mask(chm_s, np.ones_like(chm), np.zeros(chm.shape, bool), None, RES, MaskConfig())
    rows, cols = find_treetops(chm_s, mask, RES, tcfg)
    labels = segment_crowns(chm_s, mask, rows, cols, RES, SegmentConfig())
    return chm_s, mask, rows, cols, labels


def test_one_top_per_tree_including_touching_crowns():
    chm = paraboloid_chm(TREES)
    _, _, rows, cols, _ = _detect(chm)
    assert rows.size == len(TREES)
    found = sorted(zip((cols + 0.5) * RES, (rows + 0.5) * RES))
    expected = sorted((x, y) for x, y, *_ in TREES)
    for (fx, fy), (ex, ey) in zip(found, expected):
        assert np.hypot(fx - ex, fy - ey) < 1.0


def test_flat_top_gives_one_tree():
    chm = np.zeros((60, 60))
    chm[20:40, 20:40] = 10.0
    chm[28:32, 28:32] = 12.0  # flat plateau
    rows, cols = find_treetops(chm, chm > 2, RES, TreetopConfig(window_max=8))
    assert rows.size == 1


def test_crown_polygons_match_crown_size():
    chm = paraboloid_chm(TREES)
    _, _, rows, cols, labels = _detect(chm)
    grid = Grid(xmin=0, ymax=100, res=RES, width=200, height=200)
    tile = Tile("t", core=(0, 0, 100, 100), buffered=(0, 0, 100, 100))
    gdf = crowns_to_polygons(labels, chm, grid, rows, cols, tile, PolygonConfig())
    assert len(gdf) == len(TREES)
    for x, y, h, r in TREES:
        # grid y runs top-down in paraboloid_chm, so flip to map coordinates
        hit = gdf[gdf.contains(Point(x, 100 - y))]
        assert len(hit) == 1
        row = hit.iloc[0]
        assert abs(row.h_max - h) < 0.5
        # Crowns are trimmed at rel_height of the top, so the polygon is smaller than the full disk;
        # touching crowns also split the shared area.
        full = np.pi * r * r
        assert 0.3 * full < row.area_m2 <= 1.1 * full
        assert row.diam_max >= row.diam_eq * 0.9


def test_crowns_outside_core_are_dropped():
    chm = paraboloid_chm(TREES)
    _, _, rows, cols, labels = _detect(chm)
    grid = Grid(xmin=0, ymax=100, res=RES, width=200, height=200)
    tile = Tile("t", core=(0, 50, 50, 100), buffered=(0, 0, 100, 100))  # upper-left quarter
    gdf = crowns_to_polygons(labels, chm, grid, rows, cols, tile, PolygonConfig())
    # Only tree (20, 20) has its top in that quarter (map y = 80).
    assert len(gdf) == 1 and abs(gdf.iloc[0].top_x - 20) < 1


def test_mask_excludes_buildings_and_low_vegetation():
    chm = np.zeros((40, 40))
    chm[5:15, 5:15] = 8.0  # tall, porous -> tree
    chm[20:30, 20:30] = 8.0  # tall, building
    chm[5:15, 25:35] = 1.0  # low hedge
    spread = np.ones_like(chm)
    building = np.zeros_like(chm, bool)
    building[20:30, 20:30] = True
    mask = vegetation_mask(chm, spread, building, None, RES, MaskConfig())
    assert mask[10, 10] and not mask[25, 25] and not mask[10, 30]


def test_mask_uses_ndvi_when_lidar_is_not_porous():
    chm = np.zeros((40, 40))
    chm[10:30, 10:30] = 8.0
    spread = np.zeros_like(chm)  # e.g. a dense evergreen crown
    ndvi = np.full(chm.shape, -0.1)
    ndvi[10:30, 10:30] = 0.6
    building = np.zeros_like(chm, bool)
    assert vegetation_mask(chm, spread, building, ndvi, RES, MaskConfig()).any()
    assert not vegetation_mask(chm, spread, building, None, RES, MaskConfig()).any()


# -- evaluation ------------------------------------------------------------------------------

def test_match_points_is_one_to_one_and_optimal():
    a = np.array([[0, 0], [2, 0], [100, 100]], float)
    b = np.array([[1.9, 0], [0, 0], [50, 50]], float)
    ia, ib, d = match_points(a, b, 3.0)
    pairs = sorted(zip(ia.tolist(), ib.tolist()))
    assert pairs == [(0, 1), (1, 0)]  # greedy on (0 -> 1.9) would be worse
    assert np.allclose(sorted(d), [0.0, 0.1])


# -- config ----------------------------------------------------------------------------------

def test_config_rejects_unknown_keys_and_applies_overrides():
    base = {"area": {"boundary_file": "b.gpkg"}, "ahn": {"local_dir": "ahn"}}
    with pytest.raises(ValueError):
        config_from_dict({**base, "mask": {"min_hight": 2}})
    cfg = config_from_dict(base)
    c2 = cfg.with_overrides({"mask.min_height": 4.0})
    assert c2.mask.min_height == 4.0 and cfg.mask.min_height == 2.5
    with pytest.raises(KeyError):
        cfg.with_overrides({"mask.nope": 1})
