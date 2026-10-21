"""Command line: `crowns tiles|inspect|download|run|merge|evaluate|tune CONFIG`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from .config import Config, load_config
from .tiling import make_tiles, tiles_to_gdf

app = typer.Typer(add_completion=False, help="Detect tree crowns from AHN point clouds and CIR imagery.")

TilesOpt = typer.Option(None, "--tiles", "-t", help="Comma-separated tile ids (default: all)")


def _setup(config: Path, verbose: bool = False) -> Config:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pyogrio").setLevel(logging.WARNING)
    return load_config(config)


def _tiles(cfg: Config, ids: str | list[str] | None = None):
    from .sources.boundary import load_boundary

    boundary = load_boundary(cfg)
    tiles = make_tiles(boundary, cfg.tiling.size, cfg.tiling.buffer)
    if ids:
        wanted = [i.strip() for i in ids.split(",")] if isinstance(ids, str) else list(ids)
        by_id = {t.id: t for t in tiles}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise typer.BadParameter(f"Unknown tile id(s): {', '.join(missing)}. Run `crowns tiles` to list them.")
        tiles = [by_id[i] for i in wanted]
    return boundary, tiles


def _output_name(cfg: Config) -> Path:
    return cfg.output_dir / f"crowns_{cfg.project.name}.gpkg"


@app.command()
def tiles(config: Path):
    """Write the tile grid to <output>/tiles.gpkg (open it in QGIS to pick tiles)."""
    cfg = _setup(config)
    _, ts = _tiles(cfg)
    out = cfg.output_dir / "tiles.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    tiles_to_gdf(ts).to_file(out, driver="GPKG")
    typer.echo(f"{len(ts)} tiles -> {out}")


@app.command()
def inspect(laz: Path, max_points: int = 5_000_000):
    """Show the point classes in a LAZ file, to check which classes hold vegetation."""
    from .lidar import class_histogram, laz_bounds

    counts = class_histogram(laz, max_points)
    total = sum(counts.values())
    typer.echo(f"bounds: {laz_bounds(laz)}")
    for c, n in counts.items():
        typer.echo(f"class {c:>3}: {n:>12,}  ({100 * n / total:5.1f} %)")


@app.command()
def download(config: Path, tile_ids: Optional[str] = TilesOpt):
    """Download AHN sheets and imagery for the tiles, without processing."""
    from .pipeline import cir_path
    from .sources.ahn import AhnSource
    from .sources.imagery import image_for_tile

    cfg = _setup(config)
    _, ts = _tiles(cfg, tile_ids)
    paths = AhnSource(cfg).prefetch([t.buffered for t in ts])
    typer.echo(f"AHN: {len(paths)} sheets available")
    if cfg.imagery.enabled:
        from tqdm import tqdm

        for t in tqdm(ts, desc="imagery"):
            image_for_tile(cfg, t.buffered, cir_path(cfg, t.id))


@app.command()
def run(
    config: Path,
    tile_ids: Optional[str] = TilesOpt,
    workers: Optional[int] = typer.Option(None, help="Parallel processes (default: run.workers)"),
    overwrite: bool = typer.Option(False, help="Rebuild everything, ignoring cached results"),
    verbose: bool = False,
):
    """Process tiles. Without --tiles, all tiles are processed and merged into one GeoPackage."""
    from .pipeline import merge_tiles, run_tiles

    cfg = _setup(config, verbose)
    if overwrite:
        cfg = cfg.with_overrides({"run.overwrite": True})
    boundary, ts = _tiles(cfg, tile_ids)
    results = run_tiles(cfg, ts, workers)
    failed = [r for r in results if "error" in r]
    n_crowns = sum(r.get("crowns", 0) for r in results)
    typer.echo(f"{len(results) - len(failed)} tiles done, {len(failed)} failed, {n_crowns} crowns")
    for r in failed:
        typer.echo(f"  {r['tile_id']}: {r['error']}", err=True)
    if not tile_ids:
        out = _output_name(cfg)
        crowns = merge_tiles(cfg, ts, boundary, out)
        typer.echo(f"{len(crowns)} crowns -> {out}")
    if failed:
        raise typer.Exit(1)


@app.command()
def merge(config: Path):
    """Merge the per-tile results into one GeoPackage (layers: crowns, treetops)."""
    from .pipeline import merge_tiles

    cfg = _setup(config)
    boundary, ts = _tiles(cfg)
    out = _output_name(cfg)
    crowns = merge_tiles(cfg, ts, boundary, out)
    typer.echo(f"{len(crowns)} crowns -> {out}")


@app.command()
def evaluate(config: Path, tile_ids: Optional[str] = TilesOpt):
    """Compare processed tiles with the tree register; writes <output>/evaluation.gpkg/.json."""
    import geopandas as gpd
    import pandas as pd

    from .config import CRS
    from .evaluate import evaluate as run_eval
    from .evaluate import evaluated_area, load_register, load_zones, write_evaluation
    from .pipeline import tile_dir
    from .polygons import empty_crowns

    cfg = _setup(config)
    boundary, ts = _tiles(cfg, tile_ids or cfg.evaluate.tiles or None)
    ts = [t for t in ts if (tile_dir(cfg, t.id) / "done.json").exists()]
    if not ts:
        raise typer.BadParameter("None of these tiles has been processed yet; run `crowns run` first.")
    parts = [gpd.read_file(f, layer="crowns") for t in ts if (f := tile_dir(cfg, t.id) / "crowns.gpkg").exists()]
    crowns = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS) if parts else empty_crowns()
    result = run_eval(cfg, crowns, load_register(cfg), ts, evaluated_area(ts, boundary), load_zones(cfg))
    out = write_evaluation(result, cfg.output_dir, "evaluation")
    typer.echo(json.dumps(result.metrics, indent=2))
    typer.echo(f"details -> {out}")


@app.command()
def tune(config: Path, tile_ids: Optional[str] = TilesOpt):
    """Grid search over tune.grid on a few tiles; writes <output>/tuning.csv, best first."""
    from .evaluate import tune as run_tune

    cfg = _setup(config)
    ids = tile_ids or cfg.tune.tiles or cfg.evaluate.tiles
    if not ids:
        raise typer.BadParameter("Pick a few tiles with --tiles or tune.tiles; tuning on all tiles is too slow.")
    boundary, ts = _tiles(cfg, ids)
    df = run_tune(cfg, ts, boundary)
    out = cfg.output_dir / "tuning.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(df.head(10).to_string())
    typer.echo(f"all results -> {out}")


if __name__ == "__main__":
    app()
