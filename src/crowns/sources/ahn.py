"""AHN point cloud sheets: find the LAZ files that cover an area, downloading and caching them.

Three ways to point at the data (first one set wins):
- ahn.local_dir: a folder of LAZ/LAS files (sheet extents are read from the file headers).
- ahn.index_file: a GeoPackage/GeoJSON (path or URL) with one polygon per sheet and a column
  holding the download URL, e.g. the AHN6 kilometre-sheet index.
- ahn.url_template: a URL with {x_km}, {y_km} (lower-left corner in km) or {x}, {y} (metres),
  for sheets on a regular grid of ahn.template_tile_size metres.
"""

from __future__ import annotations

import hashlib
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
from shapely.geometry import box

from ..config import CRS, Config, is_url
from ..lidar import laz_bounds
from ..tiling import Bounds
from .http import download

log = logging.getLogger(__name__)


class AhnSource:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ahn = cfg.ahn
        self.cache_dir = cfg.workdir / "cache" / "ahn"
        self._index: gpd.GeoDataFrame | None = None

    # -- sheet index -------------------------------------------------------------------------

    def index(self) -> gpd.GeoDataFrame | None:
        """Sheet polygons with a 'source' column (local path or URL). None in template mode."""
        if self._index is not None:
            return self._index
        if self.ahn.local_dir:
            self._index = self._local_index()
        elif self.ahn.index_file:
            self._index = self._file_index()
        return self._index

    def _local_index(self) -> gpd.GeoDataFrame:
        folder = self.cfg.path(self.ahn.local_dir)
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".laz", ".las"))
        if not files:
            raise FileNotFoundError(f"No .laz/.las files in {folder}")
        return gpd.GeoDataFrame(
            {"source": [str(p) for p in files]},
            geometry=[box(*laz_bounds(p)) for p in files],
            crs=CRS,
        )

    def _file_index(self) -> gpd.GeoDataFrame:
        src = self.ahn.index_file
        if is_url(src):
            src = download(src, self.cfg.workdir / "cache" / _filename_from_url(src))
        else:
            src = self.cfg.path(src)
        gdf = gpd.read_file(src, layer=self.ahn.index_layer) if self.ahn.index_layer else gpd.read_file(src)
        if self.ahn.index_url_field not in gdf.columns:
            raise KeyError(
                f"Column {self.ahn.index_url_field!r} not in AHN index; available: {list(gdf.columns)}. "
                "Set ahn.index_url_field."
            )
        gdf = gdf.set_crs(CRS) if gdf.crs is None else gdf.to_crs(CRS)
        return gpd.GeoDataFrame({"source": gdf[self.ahn.index_url_field].astype(str)}, geometry=gdf.geometry, crs=CRS)

    def _template_sources(self, bounds: Bounds) -> list[str]:
        size = self.ahn.template_tile_size
        xmin, ymin, xmax, ymax = bounds
        out = []
        for i in range(math.floor(xmin / size), math.ceil(xmax / size)):
            for j in range(math.floor(ymin / size), math.ceil(ymax / size)):
                x, y = int(i * size), int(j * size)
                out.append(self.ahn.url_template.format(x=x, y=y, x_km=x // 1000, y_km=y // 1000))
        return out

    # -- files for an area ---------------------------------------------------------------------

    def sources_for(self, bounds: Bounds) -> list[str]:
        idx = self.index()
        if idx is None:
            return self._template_sources(bounds)
        hits = idx.iloc[idx.sindex.query(box(*bounds), predicate="intersects")]
        return hits["source"].tolist()

    def local_path(self, source: str) -> Path:
        return self.cache_dir / _filename_from_url(source) if is_url(source) else Path(source)

    def files_for(self, bounds: Bounds) -> list[Path]:
        """Local paths of the sheets covering `bounds`, downloading missing ones.
        Sheets that cannot be downloaded (e.g. no data over open water) are skipped with a warning."""
        paths = []
        for source in self.sources_for(bounds):
            path = self.local_path(source)
            if not path.exists() and is_url(source):
                try:
                    download(source, path)
                except RuntimeError as e:
                    log.warning("Skipping AHN sheet: %s", e)
                    continue
            if path.exists():
                paths.append(path)
        return paths

    def prefetch(self, all_bounds: list[Bounds]) -> list[Path]:
        """Download every sheet needed for a set of tiles, in parallel, before processing starts.
        Doing this up front avoids several tile workers downloading the same sheet."""
        sources = sorted({s for b in all_bounds for s in self.sources_for(b)})
        todo = [s for s in sources if is_url(s) and not self.local_path(s).exists()]

        def fetch(source: str) -> Path | None:
            try:
                return download(source, self.local_path(source))
            except RuntimeError as e:
                log.warning("Skipping AHN sheet: %s", e)
                return None

        if todo:
            log.info("Downloading %d AHN sheets", len(todo))
            with ThreadPoolExecutor(max_workers=self.ahn.download_workers) as pool:
                list(pool.map(fetch, todo))
        return [self.local_path(s) for s in sources if self.local_path(s).exists()]


def _filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name or urlparse(url).query:
        digest = hashlib.sha1(url.encode()).hexdigest()[:12]
        name = f"{digest}_{name or 'file'}"
    return name
