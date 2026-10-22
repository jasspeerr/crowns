"""Synthetic test data: a small scene with ground, trees and a building, as AHN-like LAZ plus a CIR image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import box

CRS = "EPSG:28992"
X0, Y0 = 120_000.0, 480_000.0  # lower-left corner, on a 100 m grid
SIZE = 200.0


@dataclass
class Tree:
    x: float
    y: float
    height: float
    radius: float


def default_trees() -> list[Tree]:
    trees = [
        Tree(X0 + 30, Y0 + 40, 14, 4.0),
        Tree(X0 + 60, Y0 + 150, 20, 6.0),
        Tree(X0 + 150, Y0 + 60, 10, 3.0),
        Tree(X0 + 170, Y0 + 170, 18, 5.0),
        # Two touching crowns that must stay separate.
        Tree(X0 + 120, Y0 + 130, 16, 4.5),
        Tree(X0 + 127, Y0 + 130, 13, 4.0),
        # On the 100 m tile edge: must be found exactly once.
        Tree(X0 + 100.5, Y0 + 50, 15, 4.5),
    ]
    return trees


BUILDING = (X0 + 60, Y0 + 60, X0 + 80, Y0 + 75, 9.0)  # xmin, ymin, xmax, ymax, height


def ground_z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Close to 0 m NAP, as is common in the Netherlands, with a slight slope.
    return -0.5 + 0.004 * (x - X0) + 0.002 * (y - Y0)


def canopy_surface(tree: Tree, d: np.ndarray) -> np.ndarray:
    """Height above ground of the crown surface at distance d from the stem."""
    return tree.height - 0.6 * tree.height * (d / tree.radius) ** 2


def make_points(trees: list[Tree], seed: int = 0, density: float = 10.0):
    rng = np.random.default_rng(seed)
    xs, ys, zs, cls, rn, nr = [], [], [], [], [], []

    def add(x, y, z, c, r, n):
        xs.append(x); ys.append(y); zs.append(z)
        cls.append(np.full(len(x), c, np.uint8)); rn.append(np.asarray(r, np.uint8)); nr.append(np.asarray(n, np.uint8))

    # Ground everywhere except under the building.
    n = int(SIZE * SIZE * density)
    gx = X0 + rng.uniform(0, SIZE, n)
    gy = Y0 + rng.uniform(0, SIZE, n)
    bx0, by0, bx1, by1, bh = BUILDING
    outside = ~((gx >= bx0) & (gx <= bx1) & (gy >= by0) & (gy <= by1))
    gx, gy = gx[outside], gy[outside]
    add(gx, gy, ground_z(gx, gy) + rng.normal(0, 0.03, len(gx)), 2, np.ones(len(gx)), np.ones(len(gx)))

    # Building roof: single returns, class 6.
    m = int((bx1 - bx0) * (by1 - by0) * density)
    rx = rng.uniform(bx0, bx1, m)
    ry = rng.uniform(by0, by1, m)
    add(rx, ry, ground_z(rx, ry) + bh, 6, np.ones(m), np.ones(m))

    # Tree crowns: first returns near the surface, later returns deeper in the crown.
    for t in trees:
        k = int(np.pi * t.radius ** 2 * density * 1.5)
        r = t.radius * np.sqrt(rng.uniform(0, 1, k))
        a = rng.uniform(0, 2 * np.pi, k)
        x = t.x + r * np.cos(a)
        y = t.y + r * np.sin(a)
        surface = canopy_surface(t, r)
        depth = rng.exponential(0.3, k)
        later = rng.uniform(0, 1, k) < 0.5
        depth[later] += rng.uniform(0.5, 3.0, later.sum())
        z = ground_z(x, y) + np.maximum(surface - depth, 0.5)
        add(x, y, z, 1, np.where(later, 2, 1), np.full(k, 3))

    return tuple(np.concatenate(v) for v in (xs, ys, zs, cls, rn, nr))


def write_laz(path: Path, trees: list[Tree], seed: int = 0) -> Path:
    x, y, z, c, r, n = make_points(trees, seed)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([X0, Y0, 0.0])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = c
    las.return_number = r
    las.number_of_returns = n
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)
    return path


def write_cir(path: Path, trees: list[Tree], res: float = 0.25) -> Path:
    """NIR, R, G image: crowns bright in NIR (high NDVI), everything else neutral."""
    w = h = int(SIZE / res)
    nir = np.full((h, w), 90, np.uint8)
    red = np.full((h, w), 95, np.uint8)
    green = np.full((h, w), 90, np.uint8)
    for t in trees:
        # Only touch the pixels in the tree's bounding box.
        c0 = max(int((t.x - t.radius - X0) / res), 0)
        c1 = min(int((t.x + t.radius - X0) / res) + 2, w)
        r0 = max(int((Y0 + SIZE - t.y - t.radius) / res), 0)
        r1 = min(int((Y0 + SIZE - t.y + t.radius) / res) + 2, h)
        cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        x = X0 + (cols + 0.5) * res
        y = Y0 + SIZE - (rows + 0.5) * res
        inside = (x - t.x) ** 2 + (y - t.y) ** 2 <= t.radius ** 2
        nir[rows[inside], cols[inside]] = 200
        red[rows[inside], cols[inside]] = 50
        green[rows[inside], cols[inside]] = 80
    profile = dict(driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=CRS,
                   transform=Affine(res, 0, X0, 0, -res, Y0 + SIZE))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.stack([nir, red, green]))
    return path


def write_boundary(path: Path) -> Path:
    gpd.GeoDataFrame({"name": ["test"]}, geometry=[box(X0, Y0, X0 + SIZE, Y0 + SIZE)], crs=CRS).to_file(path)
    return path


def true_crowns(trees: list[Tree], res: float = 0.25) -> gpd.GeoDataFrame:
    """Visible crown of each tree: where its canopy surface is the highest. Stands in for crowns
    a person would draw."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    n = int(SIZE / res)
    best = np.zeros((n, n))
    label = np.zeros((n, n), np.int32)
    for k, t in enumerate(trees, start=1):
        # Only the pixels in the tree's bounding box.
        c0, c1 = max(int((t.x - t.radius - X0) / res), 0), min(int((t.x + t.radius - X0) / res) + 2, n)
        r0, r1 = max(int((Y0 + SIZE - t.y - t.radius) / res), 0), min(int((Y0 + SIZE - t.y + t.radius) / res) + 2, n)
        cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        d = np.hypot(X0 + (cols + 0.5) * res - t.x, Y0 + SIZE - (rows + 0.5) * res - t.y)
        h = np.where(d <= t.radius, canopy_surface(t, d), 0)
        higher = h > best[r0:r1, c0:c1]
        best[r0:r1, c0:c1][higher] = h[higher]
        label[r0:r1, c0:c1][higher] = k
    geoms = {}
    for g, v in shapes(label, mask=label > 0, transform=Affine(res, 0, X0, 0, -res, Y0 + SIZE)):
        poly = shape(g)
        if int(v) not in geoms or poly.area > geoms[int(v)].area:
            geoms[int(v)] = poly
    keys = sorted(geoms)
    return gpd.GeoDataFrame({"tree": keys}, geometry=[geoms[k] for k in keys], crs=CRS)


def make_scene(root: Path, trees: list[Tree] | None = None) -> dict[str, Path]:
    trees = trees or default_trees()
    return {
        "laz": write_laz(root / "ahn" / "scene.laz", trees),
        "cir": write_cir(root / "cir" / "scene.tif", trees),
        "boundary": write_boundary(root / "boundary.gpkg"),
    }
