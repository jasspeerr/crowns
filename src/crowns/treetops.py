"""Tree tops: local maxima of the smoothed canopy height model, with a window that grows with height."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import disk, local_maxima

from .config import TreetopConfig


def smooth_chm(chm: np.ndarray, sigma_m: float, res: float) -> np.ndarray:
    if sigma_m <= 0:
        return chm
    return ndimage.gaussian_filter(chm, sigma=sigma_m / res)


def window_radius(height: np.ndarray, cfg: TreetopConfig) -> np.ndarray:
    """Search radius in metres for a given tree height. Taller trees have wider crowns."""
    return np.clip(cfg.window_a + cfg.window_b * height, cfg.window_min, cfg.window_max)


def find_treetops(chm_s: np.ndarray, mask: np.ndarray, res: float, cfg: TreetopConfig) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of each tree top.

    1. Candidates: regional maxima inside the mask. A flat area only counts when no
       neighbouring cell is higher, so a flat "shoulder" next to a top is not a tree.
    2. Keep a candidate only if no cell within its height-dependent radius is higher.
    3. Flat tops (equal neighbouring maxima) are reduced to one cell.
    """
    cand = local_maxima(chm_s, connectivity=2, allow_borders=True) & mask & (chm_s >= cfg.min_height)
    rows, cols = np.nonzero(cand)
    if rows.size == 0:
        return rows, cols

    heights = chm_s[rows, cols]
    radius_px = np.maximum(1, np.round(window_radius(heights, cfg) / res).astype(int))

    rmax = int(radius_px.max())
    padded = np.pad(chm_s, rmax, mode="constant", constant_values=-np.inf)
    keep = np.zeros(rows.size, dtype=bool)
    for r in np.unique(radius_px):
        sel = np.nonzero(radius_px == r)[0]
        dr, dc = np.nonzero(disk(r))
        dr = dr - r
        dc = dc - r
        # Process in batches so the (peaks x window cells) array stays small.
        batch = max(1, 2_000_000 // dr.size)
        for start in range(0, sel.size, batch):
            s = sel[start:start + batch]
            window = padded[rows[s, None] + rmax + dr[None, :], cols[s, None] + rmax + dc[None, :]]
            keep[s] = window.max(axis=1) <= heights[s]

    rows, cols = rows[keep], cols[keep]
    return _one_per_plateau(rows, cols, chm_s.shape)


def _one_per_plateau(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if rows.size < 2:
        return rows, cols
    tops = np.zeros(shape, dtype=bool)
    tops[rows, cols] = True
    labels, n = ndimage.label(tops, structure=np.ones((3, 3)))
    if n == rows.size:
        return rows, cols
    # Keep the cell nearest to the centre of each flat group.
    lab = labels[rows, cols]
    cy = ndimage.mean(rows.astype(float), lab, index=np.arange(1, n + 1))
    cx = ndimage.mean(cols.astype(float), lab, index=np.arange(1, n + 1))
    d = (rows - cy[lab - 1]) ** 2 + (cols - cx[lab - 1]) ** 2
    order = np.lexsort((d, lab))
    first = np.ones(order.size, dtype=bool)
    first[1:] = lab[order][1:] != lab[order][:-1]
    chosen = order[first]
    return rows[chosen], cols[chosen]
