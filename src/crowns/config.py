"""Configuration: a YAML file loaded into nested dataclasses.

Every value has a default, so a config file only needs the settings that differ.
Unknown keys raise an error to catch typos early.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CRS = "EPSG:28992"  # Amersfoort / RD New, the CRS of AHN and PDOK imagery


@dataclass
class ProjectConfig:
    name: str = "crowns"
    workdir: str = "data"  # cache and per-tile results
    output_dir: str = "output"


@dataclass
class AreaConfig:
    # Either a local boundary file, or a municipality code fetched from a WFS.
    boundary_file: str | None = None
    boundary_filter_field: str | None = None  # optional attribute filter for boundary_file
    boundary_filter_value: str | None = None
    municipality_code: str | None = None  # e.g. "0363" (Amsterdam)
    wfs_url: str = "https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0"
    wfs_type_name: str = "bestuurlijkegebieden:Gemeentegebied"
    wfs_code_field: str = "code"


@dataclass
class TilingConfig:
    size: float = 500.0  # tile core size in metres
    buffer: float = 25.0  # extra margin read around each core


@dataclass
class AhnConfig:
    # Pick one source. Checked in this order: local_dir, index_file, url_template.
    local_dir: str | None = None  # folder with .laz/.las files
    index_file: str | None = None  # path or URL of a GeoPackage/GeoJSON with sheet polygons
    index_layer: str | None = None
    index_url_field: str = "url"  # column in index_file holding the LAZ download URL
    url_template: str | None = None  # e.g. "https://.../AHN6/{x_km}_{y_km}.LAZ"
    template_tile_size: float = 1000.0  # sheet size for url_template, in metres
    download_workers: int = 4


@dataclass
class ImageryConfig:
    # Either a folder of georeferenced rasters, or a WMS. Set enabled: false to skip NDVI.
    enabled: bool = True
    local_dir: str | None = None
    wms_url: str = "https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0"
    wms_layer: str = "Actueel_ortho25IR"
    wms_format: str = "image/png"
    wms_max_px: int = 4000  # max width/height per GetMap request
    resolution: float = 0.25
    band_nir: int = 1  # 1-based band indices; PDOK CIR is NIR, R, G
    band_red: int = 2


@dataclass
class LidarConfig:
    resolution: float = 0.5  # raster cell size in metres
    vegetation_classes: list[int] = field(default_factory=lambda: [1])
    ground_classes: list[int] = field(default_factory=lambda: [2])
    building_classes: list[int] = field(default_factory=lambda: [6])
    roof_clearance: float = 1.0  # metres; vegetation this far above a roof is canopy, not building
    max_height: float = 60.0  # CHM values above this are treated as noise
    pit_fill_size: int = 3  # closing window (cells) used to fill holes in the CHM


@dataclass
class MaskConfig:
    min_height: float = 2.5
    use_ndvi: bool = True
    ndvi_threshold: float = 0.15
    ndvi_dilate: float = 2.0  # metres; compensates for lean of tall crowns in the orthophoto
    min_spread: float = 0.25  # share of non-last returns that marks a cell as vegetation
    building_buffer: float = 0.5  # metres
    opening_radius: float = 0.5  # metres
    min_blob_area: float = 4.0  # m²


@dataclass
class TreetopConfig:
    smooth_sigma: float = 0.5  # metres; Gaussian smoothing of the CHM before peak finding
    window_a: float = 1.0  # search radius (m) = a + b * height
    window_b: float = 0.08
    window_min: float = 1.0
    window_max: float = 6.0
    min_height: float = 3.0


@dataclass
class SegmentConfig:
    rel_height: float = 0.4  # drop crown cells lower than this fraction of the top height
    max_radius: float = 15.0  # metres from the tree top


@dataclass
class PolygonConfig:
    min_area: float = 2.0  # m²
    smooth: float = 0.25  # metres; buffer out/in to soften raster edges
    simplify: float = 0.2  # metres


@dataclass
class RunConfig:
    workers: int = 4
    save_rasters: bool = True
    overwrite: bool = False


@dataclass
class ReferenceConfig:
    """Sample plots with draft crowns that a person corrects, to serve as evaluation data."""

    file: str = "reference/reference.gpkg"
    n_plots: int = 40
    plot_size: float = 50.0  # metres
    # Plots are spread evenly over canopy-cover classes (share of the plot under crowns), so dense
    # canopy, where most errors occur, is not under-represented. Plots below the first value are skipped.
    cover_classes: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.5])
    tune_fraction: float = 0.5  # share of plots used for tuning; the rest is kept for testing
    seed: int = 0
    raster_margin: float = 10.0  # metres of background raster around each plot


@dataclass
class EvaluateConfig:
    iou_threshold: float = 0.5  # a detected crown matches a reference crown when IoU >= this
    split: str = "test"  # which plots `crowns evaluate` uses: test, tune or all


@dataclass
class TuneConfig:
    # Dotted parameter names mapped to candidate values, e.g. {"treetops.window_b": [0.06, 0.08]}.
    grid: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    area: AreaConfig = field(default_factory=AreaConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    ahn: AhnConfig = field(default_factory=AhnConfig)
    imagery: ImageryConfig = field(default_factory=ImageryConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    treetops: TreetopConfig = field(default_factory=TreetopConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    polygons: PolygonConfig = field(default_factory=PolygonConfig)
    run: RunConfig = field(default_factory=RunConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)

    # Directory of the YAML file; relative paths in the config are resolved against it.
    base_dir: Path = field(default_factory=Path.cwd, repr=False)

    def path(self, value: str | None) -> Path | None:
        """Resolve a path setting relative to the config file. URLs are not paths."""
        if value is None or is_url(value):
            return None
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def workdir(self) -> Path:
        return self.path(self.project.workdir)  # type: ignore[return-value]

    @property
    def output_dir(self) -> Path:
        return self.path(self.project.output_dir)  # type: ignore[return-value]

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Copy of this config with dotted keys replaced, e.g. {"mask.min_height": 3}."""
        new = dataclasses.replace(self)
        for key, value in overrides.items():
            section_name, _, attr = key.partition(".")
            section = getattr(new, section_name)
            if not hasattr(section, attr):
                raise KeyError(f"Unknown setting: {key}")
            setattr(new, section_name, dataclasses.replace(section, **{attr: value}))
        return new


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _build(cls: type, data: dict[str, Any] | None, where: str) -> Any:
    data = data or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown setting(s) in [{where}]: {', '.join(sorted(unknown))}")
    return cls(**data)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return config_from_dict(raw, base_dir=path.parent.resolve())


def config_from_dict(raw: dict[str, Any], base_dir: Path | None = None) -> Config:
    sections = {f.name: f for f in dataclasses.fields(Config) if f.name != "base_dir"}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"Unknown config section(s): {', '.join(sorted(unknown))}")
    kwargs: dict[str, Any] = {}
    for name, f in sections.items():
        section_cls = f.default_factory().__class__  # type: ignore[misc]
        kwargs[name] = _build(section_cls, raw.get(name), name)
    cfg = Config(**kwargs, base_dir=base_dir or Path.cwd())
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    a = cfg.area
    if not a.boundary_file and not a.municipality_code:
        raise ValueError("Set area.boundary_file or area.municipality_code")
    h = cfg.ahn
    if not (h.local_dir or h.index_file or h.url_template):
        raise ValueError("Set one of ahn.local_dir, ahn.index_file or ahn.url_template")
    if cfg.tiling.size <= 0 or cfg.tiling.buffer < 0:
        raise ValueError("tiling.size must be > 0 and tiling.buffer >= 0")
    if cfg.lidar.resolution <= 0:
        raise ValueError("lidar.resolution must be > 0")
    if cfg.evaluate.split not in ("test", "tune", "all"):
        raise ValueError("evaluate.split must be test, tune or all")
