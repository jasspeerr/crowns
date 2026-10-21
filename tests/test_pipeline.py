"""End-to-end run on a synthetic scene: LAZ + CIR image -> crowns GeoPackage -> evaluation."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from crowns.config import config_from_dict
from crowns.evaluate import evaluate, evaluated_area, load_register, load_zones, tune
from crowns.pipeline import merge_tiles, process_tile, run_tiles
from crowns.sources.boundary import load_boundary
from crowns.tiling import make_tiles
from synthetic import default_trees, make_scene


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("scene")
    paths = make_scene(root)
    cfg = config_from_dict(
        {
            "project": {"name": "test", "workdir": "work", "output_dir": "out"},
            "area": {"boundary_file": "boundary.gpkg"},
            "tiling": {"size": 100, "buffer": 20},
            "ahn": {"local_dir": "ahn"},
            "imagery": {"local_dir": "cir"},
            "run": {"workers": 1},
            "evaluate": {"register_file": "register.gpkg", "zones_file": "zones.gpkg"},
        },
        base_dir=root,
    )
    boundary = load_boundary(cfg)
    tiles = make_tiles(boundary, cfg.tiling.size, cfg.tiling.buffer)
    results = run_tiles(cfg, tiles)
    crowns = merge_tiles(cfg, tiles, boundary, cfg.output_dir / "crowns.gpkg")
    return cfg, boundary, tiles, results, crowns, paths


def test_all_tiles_succeed(scene):
    _, _, tiles, results, _, _ = scene
    assert len(tiles) == 4
    assert not [r for r in results if "error" in r]
    assert all(r["has_ndvi"] for r in results)


def test_every_tree_found_once_with_correct_height(scene):
    _, _, _, _, crowns, _ = scene
    trees = default_trees()
    assert len(crowns) == len(trees)
    for t in trees:
        hit = crowns[crowns.contains(Point(t.x, t.y))]
        assert len(hit) == 1, f"tree at {t.x - 120000:.0f},{t.y - 480000:.0f}"
        assert abs(hit.iloc[0].h_max - t.height) < 1.5
        assert np.hypot(hit.iloc[0].top_x - t.x, hit.iloc[0].top_y - t.y) < 1.5
    assert crowns["tree_id"].is_unique


def test_output_layers(scene):
    cfg, *_ = scene
    out = cfg.output_dir / "crowns.gpkg"
    tops = gpd.read_file(out, layer="treetops")
    crowns = gpd.read_file(out, layer="crowns")
    assert len(tops) == len(crowns)
    assert {"tree_id", "top_x", "top_y", "h_max", "h_mean", "area_m2", "diam_eq", "diam_max"} <= set(crowns.columns)
    assert crowns.crs.to_epsg() == 28992


def test_rerun_skips_finished_tiles_and_redetects_on_new_settings(scene):
    cfg, _, tiles, *_ = scene
    assert process_tile(cfg, tiles[0])["skipped"]
    changed = cfg.with_overrides({"segment.rel_height": 0.5})
    assert not process_tile(changed, tiles[0])["skipped"]
    assert not process_tile(cfg, tiles[0])["skipped"]  # back to the original settings


def test_evaluation_against_register(scene):
    cfg, boundary, tiles, _, crowns, _ = scene
    result = evaluate(cfg, crowns, load_register(cfg), tiles, evaluated_area(tiles, boundary), load_zones(cfg))
    m = result.metrics
    assert m["register_trees"] == len(default_trees()) + 1
    assert m["register_small_or_absent"] == 1  # the removed tree
    assert m["recall"] == 1.0
    assert m["zone_precision"] == 1.0
    assert m["merged_crowns"] == 0


def test_tune_ranks_combinations(scene):
    cfg, boundary, tiles, *_ = scene
    cfg = cfg.with_overrides({"tune.grid": {"treetops.window_b": [0.04, 0.08], "segment.rel_height": [0.4]}})
    df = tune(cfg, tiles, boundary)
    assert len(df) == 2 and df["zone_f1"].iloc[0] >= df["zone_f1"].iloc[1]
