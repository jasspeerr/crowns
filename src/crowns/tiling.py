"""Split the study area into square tiles with a buffer, and define the raster grid per tile."""

from __future__ import annotations

import math
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .config import CRS

Bounds = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


@dataclass(frozen=True)
class Tile:
    id: str
    core: Bounds  # area this tile is responsible for
    buffered: Bounds  # area read from the source data

    def owns(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """True for points inside the core. Half-open so a point on a shared edge has one owner."""
        xmin, ymin, xmax, ymax = self.core
        return (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)


def make_tiles(boundary: BaseGeometry, size: float, buffer: float) -> list[Tile]:
    """Tiles on a grid aligned to multiples of `size`, keeping those that touch the boundary."""
    xmin, ymin, xmax, ymax = boundary.bounds
    x0 = math.floor(xmin / size) * size
    y0 = math.floor(ymin / size) * size
    tiles = []
    x = x0
    while x < xmax:
        y = y0
        while y < ymax:
            core = (x, y, x + size, y + size)
            if boundary.intersects(box(*core)):
                buffered = (x - buffer, y - buffer, x + size + buffer, y + size + buffer)
                tiles.append(Tile(id=f"{int(x)}_{int(y)}", core=core, buffered=buffered))
            y += size
        x += size
    return tiles


def tiles_to_gdf(tiles: list[Tile]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"tile_id": [t.id for t in tiles]},
        geometry=[box(*t.core) for t in tiles],
        crs=CRS,
    )


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid: origin at the top-left corner."""

    xmin: float
    ymax: float
    res: float
    width: int
    height: int

    @classmethod
    def from_bounds(cls, bounds: Bounds, res: float) -> "Grid":
        xmin, ymin, xmax, ymax = bounds
        width = int(math.ceil(round((xmax - xmin) / res, 6)))
        height = int(math.ceil(round((ymax - ymin) / res, 6)))
        return cls(xmin=xmin, ymax=ymax, res=res, width=width, height=height)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0.0, self.xmin, 0.0, -self.res, self.ymax)

    @property
    def bounds(self) -> Bounds:
        return (self.xmin, self.ymax - self.height * self.res, self.xmin + self.width * self.res, self.ymax)

    def cell_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row, column and an in-grid flag for each coordinate."""
        col = np.floor((x - self.xmin) / self.res).astype(np.int64)
        row = np.floor((self.ymax - y) / self.res).astype(np.int64)
        inside = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        return row, col, inside

    def xy(self, row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates."""
        x = self.xmin + (np.asarray(col) + 0.5) * self.res
        y = self.ymax - (np.asarray(row) + 0.5) * self.res
        return x, y
