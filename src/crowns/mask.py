"""Which cells are tree canopy: tall enough, green or lidar-porous, and not a building."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import disk

from .config import MaskConfig


def vegetation_mask(
    chm: np.ndarray,
    spread: np.ndarray,
    building: np.ndarray,
    ndvi: np.ndarray | None,
    res: float,
    cfg: MaskConfig,
) -> np.ndarray:
    tall = chm >= cfg.min_height

    # Lidar evidence: the beam passes partly through foliage and branches.
    evidence = spread >= cfg.min_spread
    if cfg.use_ndvi and ndvi is not None:
        # The orthophoto is not a true ortho: tall crowns lean away from the image centre.
        # A max filter lets green pixels a few metres away still count for this cell.
        green = np.nan_to_num(ndvi, nan=-1.0)
        r = int(round(cfg.ndvi_dilate / res))
        if r > 0:
            green = ndimage.maximum_filter(green, footprint=disk(r))
        evidence |= green >= cfg.ndvi_threshold

    not_building = ~building
    rb = int(round(cfg.building_buffer / res))
    if rb > 0:
        not_building = ~ndimage.binary_dilation(building, structure=disk(rb))

    mask = tall & evidence & not_building

    ro = int(round(cfg.opening_radius / res))
    if ro > 0:
        mask = ndimage.binary_opening(mask, structure=disk(ro))
    min_cells = int(round(cfg.min_blob_area / (res * res)))
    if min_cells > 1:
        mask = remove_small_blobs(mask, min_cells)
    return mask


def remove_small_blobs(mask: np.ndarray, min_cells: int) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_cells
    keep[0] = False
    return keep[labels]
