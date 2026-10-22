"""End-to-end run on a synthetic scene: LAZ + CIR image -> crowns -> sample plots -> evaluation."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from crowns.config import config_from_dict
from crowns.evaluate import evaluate, load_reference, tune
from crowns.pipeline import merge_tiles, process_tile, run_tiles
from crowns.reference import (CROWN_LAYER, PLOT_LAYER, candidate_plots, canopy_cover, export_reference,
                              load_tile_crowns, reference_path, sample_plots)
from crowns.sources.boundary import load_boundary
from crowns.tiling import make_tiles
from synthetic import default_trees, make_scene, true_crowns


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("scene")
    make_scene(root)
    cfg = config_from_dict(
        {
            "project": {"name": "test", "workdir": "work", "output_dir": "out"},
            "area": {"boundary_file": "boundary.gpkg"},
            "tiling": {"size": 100, "buffer": 20},
            "ahn": {"local_dir": "ahn"},
            "imagery": {"local_dir": "cir"},
            "run": {"workers": 1},
            "reference": {"n_plots": 8, "plot_size": 50, "cover_classes": [0.02, 0.1]},
        },
        base_dir=root,
    )
    boundary = load_boundary(cfg)
    tiles = make_tiles(boundary, cfg.tiling.size, cfg.tiling.buffer)
    results = run_tiles(cfg, tiles)
    crowns = merge_tiles(cfg, tiles, boundary, cfg.output_dir / "crowns.gpkg")
    return cfg, boundary, tiles, results, crowns


def make_reference(scene, name, status="done", split=None, corrected=False):
    """Sample plots and simulate the person's work: mark plots done, optionally replace the drafts
    with the true crowns."""
    cfg, boundary, tiles, *_ = scene
    cfg = cfg.with_overrides({"reference.file": f"ref_{name}/reference.gpkg"})
    drafts = load_tile_crowns(cfg, tiles)
    plots = sample_plots(cfg, tiles, boundary, drafts)
    path = export_reference(cfg, plots, drafts, tiles)
    plots = gpd.read_file(path, layer=PLOT_LAYER)
    plots["status"] = status
    if split:
        plots["split"] = split
    plots.to_file(path, layer=PLOT_LAYER, driver="GPKG")
    if corrected:
        truth = true_crowns(default_trees())
        truth = gpd.GeoDataFrame({"draft_id": [None] * len(truth)}, geometry=truth.geometry, crs=truth.crs)
        truth.to_file(path, layer=CROWN_LAYER, driver="GPKG")
    return cfg, path


# -- detection ----------------------------------------------------------------------------------

def test_all_tiles_succeed(scene):
    _, _, tiles, results, _ = scene
    assert len(tiles) == 4
    assert not [r for r in results if "error" in r]
    assert all(r["has_ndvi"] for r in results)


def test_every_tree_found_once_with_correct_height(scene):
    *_, crowns = scene
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


# -- reference plots --------------------------------------------------------------------------

def test_sample_plots_export(scene):
    cfg, path = make_reference(scene, "export", status="todo")
    plots = gpd.read_file(path, layer=PLOT_LAYER)
    # Classes that run short are filled from the others, so we get 8 plots if 8 have trees.
    cand = candidate_plots(scene[2], scene[1], 50)
    eligible = int((canopy_cover(cand, load_tile_crowns(cfg, scene[2])) >= 0.02).sum())
    assert len(plots) == min(8, eligible) > 0
    assert set(plots["split"]) <= {"tune", "test"} and plots["plot_id"].is_unique
    assert (plots["cover"] >= 0.02).all()  # empty plots are skipped
    assert len(gpd.read_file(path, layer="crowns_draft")) == len(gpd.read_file(path, layer=CROWN_LAYER))
    for pid in plots["plot_id"]:
        assert (path.parent / "rasters" / f"{pid}_chm.tif").exists()
        assert (path.parent / "rasters" / f"{pid}_cir.tif").exists()
    # The reference holds hours of work: never overwrite it silently.
    with pytest.raises(FileExistsError):
        export_reference(cfg, plots, load_tile_crowns(cfg, scene[2]), scene[2])
    export_reference(cfg, plots, load_tile_crowns(cfg, scene[2]), scene[2], force=True)
    assert list(path.parent.glob("reference.backup-*.gpkg"))


def test_unfinished_plots_are_ignored(scene):
    cfg, _ = make_reference(scene, "todo", status="todo")
    with pytest.raises(ValueError, match="No plots"):
        load_reference(cfg, "all")


def test_uncorrected_draft_scores_perfect_and_is_reported_as_unchanged(scene):
    cfg, _ = make_reference(scene, "draft")
    ref = load_reference(cfg, "all")
    m = evaluate(cfg, ref, load_tile_crowns(cfg, scene[2])).metrics
    assert m["reference_crowns"] > 0
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["mean_iou_matched"] == 1.0
    assert m["edits"]["unchanged_share"] == 1.0


def test_corrected_reference(scene):
    cfg, _ = make_reference(scene, "truth", corrected=True)
    ref = load_reference(cfg, "all")
    result = evaluate(cfg, ref, load_tile_crowns(cfg, scene[2]))
    m = result.metrics
    assert m["f1"] == 1.0
    assert 0.7 < m["mean_iou_matched"] < 1.0
    assert m["merged_detections"] == 0 and m["split_references"] == 0
    assert m["edits"]["added"] == m["edits"]["reference_crowns"] and m["edits"]["unchanged"] == 0
    assert set(result.reference["status"]) <= {"matched", "outside"}


def test_missing_and_false_crowns_are_counted(scene):
    cfg, _ = make_reference(scene, "errors", corrected=True)
    ref = load_reference(cfg, "all")
    pred = load_tile_crowns(cfg, scene[2])
    scored = evaluate(cfg, ref, pred).reference
    inside = scored[scored["status"] == "matched"]
    # Remove the detection matching one reference crown: that crown becomes a miss.
    victim = inside.geometry.iloc[0]
    pred2 = pred.loc[~pred.intersects(victim.representative_point())]
    m = evaluate(cfg, ref, pred2).metrics
    assert m["matched"] == len(inside) - 1 and m["precision"] == 1.0


def test_tune_ranks_combinations(scene):
    cfg, _ = make_reference(scene, "tune", split="tune", corrected=True)
    cfg = cfg.with_overrides({"tune.grid": {"treetops.window_b": [0.04, 0.08], "segment.rel_height": [0.4]}})
    df = tune(cfg, scene[2])
    assert len(df) == 2 and df["f1"].iloc[0] >= df["f1"].iloc[1]


def test_reference_path_is_relative_to_config(scene):
    cfg, *_ = scene
    assert reference_path(cfg) == cfg.base_dir / "reference" / "reference.gpkg"
