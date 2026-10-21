"""Study area boundary: a local vector file, or a municipality fetched from a WFS (PDOK by default)."""

from __future__ import annotations

from xml.sax.saxutils import escape

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from ..config import CRS, Config
from .http import get


def load_boundary(cfg: Config) -> BaseGeometry:
    """Boundary as one (multi)polygon in EPSG:28992. WFS results are cached in the workdir."""
    a = cfg.area
    if a.boundary_file:
        gdf = gpd.read_file(cfg.path(a.boundary_file) or a.boundary_file)
        if a.boundary_filter_field:
            gdf = gdf[gdf[a.boundary_filter_field].astype(str) == str(a.boundary_filter_value)]
    else:
        cache = cfg.workdir / f"boundary_{a.municipality_code}.gpkg"
        if cache.exists():
            gdf = gpd.read_file(cache)
        else:
            gdf = fetch_municipality(a.wfs_url, a.wfs_type_name, a.wfs_code_field, str(a.municipality_code))
            cache.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(cache, driver="GPKG")
    if gdf.empty:
        raise ValueError("Boundary is empty: check the boundary file/filter or municipality code")
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS)
    return gdf.to_crs(CRS).union_all()


def fetch_municipality(wfs_url: str, type_name: str, code_field: str, code: str) -> gpd.GeoDataFrame:
    fes = (
        '<fes:Filter xmlns:fes="http://www.opengis.net/fes/2.0">'
        f"<fes:PropertyIsEqualTo><fes:ValueReference>{escape(code_field)}</fes:ValueReference>"
        f"<fes:Literal>{escape(code)}</fes:Literal></fes:PropertyIsEqualTo></fes:Filter>"
    )
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": type_name,
        "outputFormat": "application/json",
        "srsName": CRS,
        "filter": fes,
    }
    data = get(wfs_url, params=params).json()
    features = data.get("features", [])
    if not features:
        raise ValueError(f"No feature with {code_field}={code!r} in {type_name} at {wfs_url}")
    return gpd.GeoDataFrame.from_features(features, crs=CRS)
