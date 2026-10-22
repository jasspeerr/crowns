# crowns

Detect individual tree crowns as polygons for a whole municipality, from
**AHN point clouds** (height) and **PDOK 25 cm CIR aerial imagery** (near-infrared).
Classical method, no training data or GPU needed.

Output: `output/crowns_<name>.gpkg` with two layers:

| layer      | geometry | attributes |
|------------|----------|------------|
| `crowns`   | polygon per crown | `tree_id`, `tile_id`, `top_x`, `top_y`, `h_max`, `h_mean` (m above ground), `area_m2`, `diam_eq` (diameter of a circle with the same area), `diam_max` (longest distance across the crown) |
| `treetops` | point per tree top | same attributes |

## How it works

The area is split into 500 m tiles, each read with a 25 m buffer so crowns on tile edges are
complete. A crown belongs to the tile that contains its tree top, so every tree appears once.

Per tile:

1. **Rasters from the point cloud (0.5 m):** terrain (lowest ground point, gaps interpolated),
   canopy height model (CHM = highest vegetation point minus terrain; empty cells inside crowns
   filled), building mask, and *return spread* (share of vegetation points that are not the last
   return: high in foliage, low on roofs and cars).
2. **NDVI** from the CIR image, averaged onto the same grid. The orthophoto is not a true ortho, so
   tall crowns can appear shifted by a few metres. NDVI is therefore only supporting evidence
   (dilated by `mask.ndvi_dilate`), and crown shapes always come from the CHM.
3. **Vegetation mask:** CHM ≥ `mask.min_height`, AND (NDVI ≥ threshold OR return spread ≥ threshold),
   AND not a building. Small blobs are removed.
4. **Tree tops:** regional maxima of the smoothed CHM, kept only if nothing is higher within a
   radius that grows with height (`window_a + window_b × height`).
5. **Crowns:** marker-controlled watershed on the CHM from the tree tops. Cells lower than
   `segment.rel_height` × top height, or further than `segment.max_radius`, are removed.
6. **Polygons:** vectorised, holes removed, lightly smoothed and simplified.

Results are cached per tile in `data/tiles/<tile_id>/`. This includes `chm.tif`, `dtm.tif`,
`ndvi.tif`, `mask.tif` and `crowns.tif` for checking in QGIS. If you change only detection settings
(`mask`, `treetops`, `segment`, `polygons`), a re-run reuses the rasters and only redoes detection,
which takes seconds per tile.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # runs on synthetic data, no downloads needed
```

## Use

1. Copy `config/example.yaml` and set the municipality, data sources and paths.
2. Check the AHN classes in one downloaded sheet. AHN normally puts vegetation in class 1
   ("unclassified"); adjust `lidar.vegetation_classes` if AHN6 differs:
   ```bash
   crowns inspect data/cache/ahn/<sheet>.LAZ
   ```
3. Run everything. Without `--tiles`, all results are merged into one GeoPackage:
   ```bash
   crowns run my.yaml --workers 4
   ```
   To try a few tiles first: `crowns tiles my.yaml` writes the grid to `output/tiles.gpkg`, then
   `crowns run my.yaml --tiles 136000_455500,136500_455500`.
4. Make evaluation data by correcting sample plots, then tune and evaluate (see below).

`crowns download my.yaml` fetches all data first without processing, which is useful before going
offline. If tiles fail, the run reports them; running again only redoes unfinished tiles.

### Data sources

- **Boundary:** `area.municipality_code` (PDOK *Bestuurlijke gebieden* WFS), or `area.boundary_file`.
- **AHN:** one of
  - `ahn.index_file`: GeoPackage/GeoJSON of AHN sheets with a download-URL column (AHN6 uses
    1 × 1 km sheets; its index is published as a GeoPackage). Set `ahn.index_url_field` to that column.
  - `ahn.url_template`: e.g. `https://…/{x_km}_{y_km}.LAZ` for a regular sheet grid.
  - `ahn.local_dir`: LAZ files you already have.

  Sheets are downloaded once to `data/cache/ahn/`.
- **Imagery:** the PDOK CIR WMS by default, requested per tile as lossless PNG at 0.25 m.
  Or point `imagery.local_dir` at GeoTIFF/JP2 files. Pick a year (`imagery.wms_layer`) close to the
  AHN flight. Set `imagery.enabled: false` to use lidar evidence only.

> The default PDOK URLs and the WMS layer name (`Actueel_ortho25IR`) could not be tested while this
> was built, because PDOK wasn't reachable from the build machine. If a request fails, check the
> layer names in the WMS GetCapabilities and the field names of the boundary WFS, and adjust the config.

## Evaluation data: correcting draft results

The evaluation data is made by correcting the pipeline's own draft results in sample plots,
which is much faster than drawing crowns from scratch.

```bash
crowns sample-plots my.yaml        # picks plots, writes reference/reference.gpkg + rasters
# ... correct the plots in QGIS ...
crowns tune my.yaml                # tries tune.grid on the 'tune' plots -> output/tuning.csv
crowns run my.yaml                 # re-run with the chosen settings (only detection is redone)
crowns evaluate my.yaml            # scores the 'test' plots -> output/evaluation.json/.gpkg
```

**Sampling.** `sample-plots` picks `reference.n_plots` squares of `reference.plot_size` m (default
40 × 50 m) at random from processed tiles. The plots are spread evenly over canopy-cover classes, so
dense canopy, where most errors happen, is well represented. If a class runs short, the rest is
filled from the other classes. Half the plots are marked `tune`, the other half `test`.

**Correcting.** Open `reference/reference.gpkg` in QGIS with the background rasters in
`reference/rasters/` (`P001_chm.tif` canopy height, `P001_cir.tif` infrared photo). For each plot:

1. Edit the `crowns` layer for every crown whose **centre lies inside the plot**:
   delete false crowns (hedges, poles, roofs), draw missing trees, fix outlines, and split or merge
   crowns. Crowns crossing the plot edge are included so they can be fixed too.
2. Set the plot's `status` to `done` only after every tree in it has been checked. Plots that are
   not `done` are ignored, so you can work in several sessions.

**Avoiding a flattering reference.** When correcting a draft, it is easy to accept outlines that
look roughly right and to overlook trees the model also missed. That makes the scores too
optimistic. To limit this:
- Scan every plot systematically for missing trees (CHM and photo side by side), don't only check
  the drafts.
- `crowns_draft` keeps an untouched copy. `evaluate` reports under `edits` how many crowns were
  unchanged, reshaped, added or deleted. A very high unchanged share is a warning sign.
- Keep the reference fixed once made. `sample-plots` refuses to overwrite it (`--force` keeps a
  backup). Later model versions are then judged against the same data. Avoid re-drafting plots
  from a newer model.
- Tune only on `tune` plots and report results on `test` plots.

**Scores** (`output/evaluation.json`):
- Detected and reference crowns are paired one-to-one by overlap (IoU = intersection / union),
  with an optimal (Hungarian) assignment; a pair matches when IoU ≥ `evaluate.iou_threshold` (0.5).
- `precision`: share of detected crowns that are real. `recall`: share of reference crowns found.
  `f1` combines both. `mean_iou_matched` measures how well the matched outlines fit.
- `merged_detections`: detected crowns covering two or more reference crowns (under-segmentation).
  `split_references`: reference crowns split into two or more detections (over-segmentation).
- `canopy_area_precision` / `canopy_area_recall`: total canopy area agreement, ignoring individual trees.
- `recall_by_size` and `by_cover_class`: where the errors are.
- `output/evaluation.gpkg` has the plots (with scores per plot), the reference crowns and the
  detections, each with a `status` column (matched / missed / false), to find problems in QGIS.

`crowns tune` tries every combination in `tune.grid` (detection settings only) on the finished
`tune` plots. It writes `output/tuning.csv`, best F1 first.

## Performance

Tested on synthetic data with AHN-like density (~25 points/m²): a 500 m tile (8 million points)
takes about 10 s and under 1 GB of memory. For ~100 km² (~400 tiles) on 4 cores, expect
10–20 minutes plus download time.

## Known limitations

- Small trees under or squeezed between larger crowns leave no peak in the height model and are
  missed or merged with their neighbour. This is inherent to height-model methods.
- AHN is often flown leaf-off, so deciduous crowns can be sparse. Empty cells inside crowns are filled,
  but very open crowns may still be split into several trees. Tune `treetops.smooth_sigma` and
  `segment.rel_height`.
- Imagery and lidar from different years: trees felled or planted in between cause mismatches.
- Crowns hanging over roofs are cut at the building outline (`mask.building_buffer`).
