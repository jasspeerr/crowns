"""Grow a crown around each tree top with a marker-controlled watershed on the canopy height model."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from .config import SegmentConfig


def segment_crowns(
    chm_s: np.ndarray,
    mask: np.ndarray,
    top_rows: np.ndarray,
    top_cols: np.ndarray,
    res: float,
    cfg: SegmentConfig,
) -> np.ndarray:
    """Label raster: 0 = no crown, k = crown of tree top k-1 (same order as the input tops)."""
    labels = np.zeros(chm_s.shape, dtype=np.int32)
    n = top_rows.size
    if n == 0:
        return labels

    markers = np.zeros(chm_s.shape, dtype=np.int32)
    markers[top_rows, top_cols] = np.arange(1, n + 1, dtype=np.int32)
    # Flood "downhill" from each top: watershed on the inverted height model.
    labels = watershed(-chm_s, markers=markers, mask=mask)

    # Remove crown-edge spill: cells much lower than their own tree top (e.g. hedges or
    # lower vegetation that touches the crown).
    top_h = np.concatenate([[np.inf], chm_s[top_rows, top_cols]])
    labels[chm_s < cfg.rel_height * top_h[labels]] = 0

    # Remove cells too far from their tree top.
    if cfg.max_radius > 0:
        rr, cc = np.nonzero(labels)
        lab = labels[rr, cc]
        d2 = (rr - top_rows[lab - 1]) ** 2 + (cc - top_cols[lab - 1]) ** 2
        far = d2 > (cfg.max_radius / res) ** 2
        labels[rr[far], cc[far]] = 0

    return _keep_part_with_top(labels, top_rows, top_cols)


def _keep_part_with_top(labels: np.ndarray, top_rows: np.ndarray, top_cols: np.ndarray) -> np.ndarray:
    """After trimming, a crown can fall apart; keep only the piece that contains its tree top."""
    out = np.zeros_like(labels)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        crown = labels[sl] == k
        pieces, n = ndimage.label(crown)
        if n == 1:
            out[sl][crown] = k
            continue
        r, c = top_rows[k - 1] - sl[0].start, top_cols[k - 1] - sl[1].start
        if 0 <= r < crown.shape[0] and 0 <= c < crown.shape[1] and pieces[r, c] > 0:
            out[sl][pieces == pieces[r, c]] = k
    return out
