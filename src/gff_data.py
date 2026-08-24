"""Dataset utilities for GFF 24/48/72-hour flood-footprint hindcasts."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
import sqlite3
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import cv2
import rasterio
import torch
import xarray as xr
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from torch.nn import functional as F
from torch.utils.data import Dataset


ERA5_BANDS = (
    "mean_2m_air_temperature",
    "minimum_2m_air_temperature",
    "maximum_2m_air_temperature",
    "dewpoint_2m_temperature",
    "total_precipitation",
    "surface_pressure",
    "mean_sea_level_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
)
ERA5_LAND_BANDS = (
    "dewpoint_temperature_2m",
    "temperature_2m",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "surface_net_solar_radiation_sum",
    "surface_net_thermal_radiation_sum",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_pressure",
    "total_precipitation_sum",
    "snow_depth_water_equivalent",
    "potential_evaporation_sum",
)
GLOFAS_BANDS = ("dis24", "rowe", "swir")
WEATHER_CHANNELS = len(ERA5_BANDS) + len(ERA5_LAND_BANDS) + len(GLOFAS_BANDS)


@dataclass(frozen=True)
class GFFTile:
    meta_path: Path
    bounds: tuple[float, float, float, float]
    n_background: int
    n_permanent_water: int
    n_flooded: int

    @property
    def is_flooded(self) -> bool:
        return self.n_flooded > 0


def _normalisation(path: Path) -> dict[str, tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["band"]: (float(row["mean"]), max(float(row["std"]), 1e-6))
            for row in csv.DictReader(handle)
        }


def _feature_table(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT table_name FROM gpkg_contents WHERE data_type='features' LIMIT 1"
    ).fetchone()
    if row is None:
        raise ValueError("GeoPackage contains no feature table")
    return str(row[0])


def _gpkg_geometry_bounds(blob: bytes) -> tuple[float, float, float, float]:
    """Read the envelope in a standard GeoPackage geometry header."""

    if len(blob) < 40 or blob[:2] != b"GP":
        raise ValueError("Unsupported GeoPackage geometry header")
    flags = blob[3]
    endian = "<" if flags & 1 else ">"
    envelope_type = (flags >> 1) & 0b111
    if envelope_type == 0:
        raise ValueError("GeoPackage geometry has no cached envelope")
    min_x, max_x, min_y, max_y = struct.unpack(f"{endian}4d", blob[8:40])
    return min_x, min_y, max_x, max_y


def read_tile_index(path: Path) -> list[tuple[tuple[float, float, float, float], int, int, int]]:
    connection = sqlite3.connect(path)
    try:
        table = _feature_table(connection).replace('"', '""')
        rows = connection.execute(
            f'SELECT geom, n_background, n_permanent_water, n_flooded FROM "{table}"'
        )
        return [
            (
                _gpkg_geometry_bounds(bytes(geometry)),
                int(background),
                int(permanent),
                int(flooded),
            )
            for geometry, background, permanent, flooded in rows
        ]
    finally:
        connection.close()


def mercator_bounds_to_lonlat(
    bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Convert EPSG:3857 bounds to longitude/latitude without a GIS dependency."""

    radius = 6_378_137.0

    def convert(x: float, y: float) -> tuple[float, float]:
        lon = math.degrees(x / radius)
        lat = math.degrees(2.0 * math.atan(math.exp(y / radius)) - math.pi / 2.0)
        return lon, lat

    min_lon, min_lat = convert(bounds[0], bounds[1])
    max_lon, max_lat = convert(bounds[2], bounds[3])
    return min_lon, min_lat, max_lon, max_lat


def _read_raster_tile(
    path: Path,
    bounds: tuple[float, float, float, float],
    bands: Iterable[int],
    size: int,
    resampling: Resampling,
) -> np.ndarray:
    band_indexes = tuple(bands)
    with rasterio.open(path) as source:
        window = from_bounds(*bounds, transform=source.transform)
        values = source.read(
            list(band_indexes),
            window=window,
            out_shape=(len(band_indexes), size, size),
            boundless=True,
            fill_value=source.nodata,
            resampling=resampling,
        )
        if source.nodata is not None:
            values = values.astype(np.float32, copy=False)
            values[values == source.nodata] = np.nan
    return values


def sunet_clahe_sar(
    values: np.ndarray,
    db_ranges: tuple[tuple[float, float], tuple[float, float]] = (
        (-25.0, 0.0),
        (-32.0, -5.0),
    ),
    clip_limit: float = 2.0,
    grid_size: int = 8,
    enhancement_size: int = 256,
) -> np.ndarray:
    """Adapt SU-Net's original-plus-CLAHE enhancement to Sentinel-1 VV/VH.

    SU-Net operates on display-like ISAR intensities. GFF stores linear SAR
    backscatter, so the physically meaningful logarithmic conversion is done
    first. Both the clipped dB image and its CLAHE version are returned in
    ``[-1, 1]`` as ``VV, VH, VV-CLAHE, VH-CLAHE``.
    """

    if values.shape[0] != 2:
        raise ValueError(f"Expected VV/VH SAR with two bands, got {values.shape}")
    if grid_size < 1 or enhancement_size < 1 or clip_limit <= 0:
        raise ValueError("CLAHE parameters must be positive")
    height, width = values.shape[-2:]
    original: list[np.ndarray] = []
    enhanced: list[np.ndarray] = []
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(int(grid_size), int(grid_size))
    )
    for band, (low, high) in zip(values, db_ranges):
        if high <= low:
            raise ValueError(f"Invalid SAR dB range: {(low, high)}")
        linear = np.nan_to_num(
            band.astype(np.float32, copy=False), nan=1e-6, posinf=1.0, neginf=1e-6
        )
        db = 10.0 * np.log10(np.maximum(linear, 1e-6))
        unit = np.clip((db - low) / (high - low), 0.0, 1.0).astype(np.float32)
        original.append(unit * 2.0 - 1.0)
        image = np.rint(unit * 255.0).astype(np.uint8)
        if (height, width) != (enhancement_size, enhancement_size):
            image = cv2.resize(
                image,
                (enhancement_size, enhancement_size),
                interpolation=cv2.INTER_LINEAR,
            )
        image = clahe.apply(image)
        if image.shape != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        enhanced.append(image.astype(np.float32) / 127.5 - 1.0)
    return np.stack([*original, *enhanced]).astype(np.float32)


def sunet_sar_variant(
    values: np.ndarray,
    mode: str,
    db_ranges: tuple[tuple[float, float], tuple[float, float]] = (
        (-25.0, 0.0),
        (-32.0, -5.0),
    ),
    clip_limit: float = 2.0,
    grid_size: int = 8,
    enhancement_size: int = 256,
) -> np.ndarray:
    """Select a controlled SU-Net preprocessing ablation for VV/VH."""

    original_enhanced = sunet_clahe_sar(
        values,
        db_ranges=db_ranges,
        clip_limit=clip_limit,
        grid_size=grid_size,
        enhancement_size=enhancement_size,
    )
    if mode == "sunet_db":
        return original_enhanced[:2]
    if mode == "sunet_clahe_only":
        return original_enhanced[2:]
    if mode == "sunet_clahe":
        return original_enhanced
    raise ValueError(f"Unknown SU-Net SAR preprocessing mode: {mode}")


class _XarrayCache:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.datasets: OrderedDict[Path, xr.Dataset] = OrderedDict()

    def get(self, path: Path) -> xr.Dataset:
        dataset = self.datasets.pop(path, None)
        if dataset is None:
            dataset = xr.open_dataset(path, engine="h5netcdf")
        self.datasets[path] = dataset
        while len(self.datasets) > self.capacity:
            _, old = self.datasets.popitem(last=False)
            old.close()
        return dataset

    def close(self) -> None:
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()


class GFFFloodForecastDataset(Dataset):
    """Tile-level GFF samples replicated for 1/2/3-day horizons.

    GFF provides reanalysis through the target day, not archived operational
    forecasts. In the default causal mode, the final ``horizon`` days are
    masked to the normalized climatological mean so no post-issue observation
    leaks into the 24/48/72-hour prediction. ``perfect`` mode is an optional
    upper bound that treats those reanalysis days as error-free forecasts.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        fold: int = 0,
        horizons: tuple[int, ...] = (1, 2, 3),
        weather_window: int = 20,
        tile_size: int = 224,
        context_size: int = 16,
        context_buffer_m: float = 50_000.0,
        max_tiles: int | None = None,
        max_sites: int | None = None,
        seed: int = 1337,
        augment: bool = False,
        strict: bool = True,
        forcing_mode: str = "causal",
        sar_preprocessing: str = "standard",
        sar_db_ranges: tuple[tuple[float, float], tuple[float, float]] = (
            (-25.0, 0.0),
            (-32.0, -5.0),
        ),
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        clahe_enhancement_size: int = 256,
    ):
        self.root = Path(root)
        self.rois = self.root / "rois"
        self.split = split
        self.fold = int(fold)
        self.horizons = tuple(int(value) for value in horizons)
        self.weather_window = int(weather_window)
        self.tile_size = int(tile_size)
        self.context_size = int(context_size)
        self.context_buffer_m = float(context_buffer_m)
        self.augment = bool(augment)
        self.strict = bool(strict)
        self.forcing_mode = str(forcing_mode).lower()
        self.sar_preprocessing = str(sar_preprocessing).lower()
        self.sar_db_ranges = tuple(tuple(float(item) for item in pair) for pair in sar_db_ranges)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_grid_size = int(clahe_grid_size)
        self.clahe_enhancement_size = int(clahe_enhancement_size)
        self.sar_channel_count = 4 if self.sar_preprocessing == "sunet_clahe" else 2
        self.spatial_channels = self.sar_channel_count + 2
        self._cache = _XarrayCache()
        self._static_tensor_cache: OrderedDict[
            GFFTile, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self._weather_tensor_cache: OrderedDict[GFFTile, torch.Tensor] = OrderedDict()

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if self.forcing_mode not in {"causal", "perfect"}:
            raise ValueError("forcing_mode must be 'causal' or 'perfect'")
        valid_sar_preprocessing = {
            "standard",
            "sunet_db",
            "sunet_clahe_only",
            "sunet_clahe",
        }
        if self.sar_preprocessing not in valid_sar_preprocessing:
            choices = ", ".join(sorted(valid_sar_preprocessing))
            raise ValueError(f"sar_preprocessing must be one of: {choices}")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("Horizons must contain positive day counts")
        if max(self.horizons) >= self.weather_window:
            raise ValueError("Every horizon must be shorter than weather_window")

        norm_root = self.root / "normalisation"
        self.s1_norm = _normalisation(norm_root / f"s1_norm_{fold}.csv")
        self.dem_norm = _normalisation(norm_root / f"dem_norm_{fold}.csv")["dem"]
        self.hand_norm = _normalisation(norm_root / f"hand_norm_{fold}.csv")["hand"]
        self.era5_norm = _normalisation(norm_root / "era5_norm.csv")
        self.era5_land_norm = _normalisation(norm_root / "era5_land_norm.csv")
        self.glofas_norm = _normalisation(norm_root / f"glofas_norm_{fold}.csv")

        # Partition lists also mention the optional ``extras.zip`` sites that
        # fall outside ERA5 availability. They cannot support this task and are
        # intentionally excluded when only the core forecasting components are
        # downloaded.
        site_names = [
            name for name in self._split_sites(split) if (self.rois / Path(name).name).exists()
        ]
        if max_sites is not None:
            site_names = site_names[: int(max_sites)]
        tiles: list[GFFTile] = []
        missing: list[str] = []
        for name in site_names:
            meta_path = self.rois / Path(name).name
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            paths = self.component_paths(meta_path, metadata)
            absent = [str(path) for path in paths.values() if not path.exists()]
            if absent:
                missing.extend(absent)
                continue
            for bounds, background, permanent, flooded in read_tile_index(paths["geometry"]):
                tiles.append(
                    GFFTile(meta_path, bounds, background, permanent, flooded)
                )
        if missing and strict:
            preview = "\n".join(missing[:8])
            raise FileNotFoundError(
                f"{len(missing)} required GFF component files are missing. First paths:\n{preview}"
            )
        rng = random.Random(seed + {"train": 0, "val": 1, "test": 2}[split])
        rng.shuffle(tiles)
        if max_tiles is not None:
            tiles = tiles[: int(max_tiles)]
        if not tiles:
            raise RuntimeError(
                f"No complete GFF tiles found for split={split}; finish component downloads first"
            )
        self.tiles = tiles

    def _split_sites(self, split: str) -> list[str]:
        test_fold = self.fold
        val_fold = (self.fold + 1) % 5
        if split == "test":
            folds = [test_fold]
        elif split == "val":
            folds = [val_fold]
        else:
            folds = [value for value in range(5) if value not in {test_fold, val_fold}]
        names: list[str] = []
        for value in folds:
            path = self.root / "partitions" / f"floodmap_partition_{value}.txt"
            names.extend(
                line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        return names

    @staticmethod
    def component_paths(meta_path: Path, metadata: dict) -> dict[str, Path]:
        rois = meta_path.parent
        stem = meta_path.name.removesuffix("-meta.json")
        pre_date = str(metadata["pre1_date"])[:10]
        post_date = str(metadata["post_date"])[:10]
        return {
            "geometry": rois / metadata["visit_tiles"],
            "target": rois / metadata["floodmap"],
            "s1": rois / f"{metadata['key']}-{pre_date}-s1.tif",
            "dem": rois / f"{stem}-dem-local.tif",
            "hand": rois / f"{stem}-hand.tif",
            "era5": rois / f"{stem}-era5.nc",
            "era5_land": rois / f"{stem}-era5-land.nc",
            "glofas": rois / f"{metadata['key']}_{post_date}.nc",
        }

    @property
    def tile_positive_flags(self) -> list[bool]:
        return [tile.is_flooded for tile in self.tiles]

    @property
    def sample_positive_flags(self) -> list[bool]:
        return [tile.is_flooded for tile in self.tiles for _ in self.horizons]

    def __len__(self) -> int:
        return len(self.tiles) * len(self.horizons)

    def _dynamic_source(
        self,
        path: Path,
        bands: tuple[str, ...],
        normalisation: dict[str, tuple[float, float]],
        lonlat_bounds: tuple[float, float, float, float],
        post_date: dt.datetime,
    ) -> torch.Tensor:
        dataset = self._cache.get(path)
        min_lon, min_lat, max_lon, max_lat = lonlat_bounds
        longitude = dataset.longitude.values
        if float(np.nanmin(longitude)) > 180.0:
            min_lon += 360.0
            max_lon += 360.0

        lon_slice = (
            slice(min_lon, max_lon)
            if longitude[0] <= longitude[-1]
            else slice(max_lon, min_lon)
        )
        latitude = dataset.latitude.values
        lat_slice = (
            slice(min_lat, max_lat)
            if latitude[0] <= latitude[-1]
            else slice(max_lat, min_lat)
        )
        subset = dataset.sel(longitude=lon_slice, latitude=lat_slice)
        if subset.sizes.get("longitude", 0) == 0 or subset.sizes.get("latitude", 0) == 0:
            subset = dataset.sel(
                longitude=[(min_lon + max_lon) / 2.0],
                latitude=[(min_lat + max_lat) / 2.0],
                method="nearest",
            )

        dates = np.array(
            [
                np.datetime64((post_date - dt.timedelta(days=offset)).date())
                for offset in range(self.weather_window - 1, -1, -1)
            ]
        )
        subset = subset.sel(time=dates, method="nearest")
        arrays = []
        for band in bands:
            values = subset[band].transpose("time", "latitude", "longitude").values
            mean, std = normalisation[band]
            arrays.append((values.astype(np.float32) - mean) / std)
        stacked = np.stack(arrays, axis=1)
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=6.0, neginf=-6.0)
        stacked = np.clip(stacked, -6.0, 6.0)
        return F.interpolate(
            torch.from_numpy(stacked),
            size=(self.context_size, self.context_size),
            mode="bilinear",
            align_corners=False,
        )

    def _static_components(self, tile: GFFTile) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._static_tensor_cache.pop(tile, None)
        if cached is not None:
            self._static_tensor_cache[tile] = cached
            return cached
        metadata = json.loads(tile.meta_path.read_text(encoding="utf-8"))
        paths = self.component_paths(tile.meta_path, metadata)
        target_raw = _read_raster_tile(
            paths["target"], tile.bounds, (1,), self.tile_size, Resampling.nearest
        )[0]
        valid = np.isfinite(target_raw) & (target_raw <= 2)
        target = target_raw == 2

        s1 = _read_raster_tile(
            paths["s1"], tile.bounds, (1, 2), self.tile_size, Resampling.bilinear
        ).astype(np.float32)
        if self.sar_preprocessing != "standard":
            s1 = sunet_sar_variant(
                s1,
                mode=self.sar_preprocessing,
                db_ranges=self.sar_db_ranges,
                clip_limit=self.clahe_clip_limit,
                grid_size=self.clahe_grid_size,
                enhancement_size=self.clahe_enhancement_size,
            )
        else:
            for index, band in enumerate(("VV", "VH")):
                mean, std = self.s1_norm[band]
                s1[index] = (s1[index] - mean) / std
        dem = _read_raster_tile(
            paths["dem"], tile.bounds, (1,), self.tile_size, Resampling.bilinear
        ).astype(np.float32)
        hand = _read_raster_tile(
            paths["hand"], tile.bounds, (1,), self.tile_size, Resampling.bilinear
        ).astype(np.float32)
        dem = (dem - self.dem_norm[0]) / self.dem_norm[1]
        hand = (hand - self.hand_norm[0]) / self.hand_norm[1]
        spatial = np.concatenate([s1, dem, hand], axis=0)
        spatial = np.nan_to_num(spatial, nan=0.0, posinf=6.0, neginf=-6.0)
        spatial = torch.from_numpy(np.clip(spatial, -6.0, 6.0))

        target_tensor = torch.from_numpy(target.astype(np.float32))[None]
        valid_tensor = torch.from_numpy(valid.astype(np.float32))[None]
        result = spatial.float(), target_tensor, valid_tensor
        self._static_tensor_cache[tile] = result
        while len(self._static_tensor_cache) > 24:
            self._static_tensor_cache.popitem(last=False)
        return result

    def _weather_components(self, tile: GFFTile) -> torch.Tensor:
        cached = self._weather_tensor_cache.pop(tile, None)
        if cached is not None:
            self._weather_tensor_cache[tile] = cached
            return cached
        metadata = json.loads(tile.meta_path.read_text(encoding="utf-8"))
        paths = self.component_paths(tile.meta_path, metadata)
        post_date = dt.datetime.fromisoformat(str(metadata["post_date"]))
        min_x, min_y, max_x, max_y = tile.bounds
        buffer_m = self.context_buffer_m
        lonlat = mercator_bounds_to_lonlat(
            (min_x - buffer_m, min_y - buffer_m, max_x + buffer_m, max_y + buffer_m)
        )
        result = torch.cat(
            [
                self._dynamic_source(
                    paths["era5"], ERA5_BANDS, self.era5_norm, lonlat, post_date
                ),
                self._dynamic_source(
                    paths["era5_land"],
                    ERA5_LAND_BANDS,
                    self.era5_land_norm,
                    lonlat,
                    post_date,
                ),
                self._dynamic_source(
                    paths["glofas"], GLOFAS_BANDS, self.glofas_norm, lonlat, post_date
                ),
            ],
            dim=1,
        ).float()
        self._weather_tensor_cache[tile] = result
        while len(self._weather_tensor_cache) > 64:
            self._weather_tensor_cache.popitem(last=False)
        return result

    def _load_sample(self, tile: GFFTile, horizon: int) -> dict:
        spatial, target_tensor, valid_tensor = self._static_components(tile)
        weather = self._weather_components(tile)
        # Augmentation mutates its inputs, while cached tensors must remain immutable.
        spatial = spatial.clone()
        target_tensor = target_tensor.clone()
        valid_tensor = valid_tensor.clone()
        weather = weather.clone()
        forecast_mask = torch.zeros(self.weather_window, dtype=torch.bool)
        forecast_mask[-horizon:] = True
        if self.forcing_mode == "causal":
            # Strict fixed-horizon forecast: no post-issue reanalysis is visible.
            # Zero is the normalized climatological mean for every channel.
            weather[-horizon:] = 0.0

        if self.augment:
            if random.random() < 0.5:
                spatial = spatial.flip(-1)
                weather = weather.flip(-1)
                target_tensor = target_tensor.flip(-1)
                valid_tensor = valid_tensor.flip(-1)
            if random.random() < 0.5:
                spatial = spatial.flip(-2)
                weather = weather.flip(-2)
                target_tensor = target_tensor.flip(-2)
                valid_tensor = valid_tensor.flip(-2)
            rotations = random.randrange(4)
            if rotations:
                spatial = torch.rot90(spatial, rotations, (-2, -1))
                weather = torch.rot90(weather, rotations, (-2, -1))
                target_tensor = torch.rot90(target_tensor, rotations, (-2, -1))
                valid_tensor = torch.rot90(valid_tensor, rotations, (-2, -1))
            if random.random() < 0.5:
                gain = random.uniform(0.9, 1.1)
                sar = spatial[: self.sar_channel_count]
                spatial[: self.sar_channel_count] = sar * gain + torch.randn_like(sar) * 0.03

        dilated = F.max_pool2d(target_tensor, 5, stride=1, padding=2)
        eroded = -F.max_pool2d(-target_tensor, 5, stride=1, padding=2)
        boundary = (dilated - eroded).clamp(0.0, 1.0) * valid_tensor
        return {
            "spatial": spatial.float(),
            "weather": weather.float(),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "forecast_mask": forecast_mask,
            "target": target_tensor,
            "valid": valid_tensor,
            "boundary": boundary,
            "presence": torch.tensor(float(target_tensor.any()), dtype=torch.float32),
            "site": tile.meta_path.name.removesuffix("-meta.json"),
            "bounds": torch.tensor(tile.bounds, dtype=torch.float64),
        }

    def __getitem__(self, index: int) -> dict:
        tile_index, horizon_index = divmod(index, len(self.horizons))
        return self._load_sample(self.tiles[tile_index], self.horizons[horizon_index])

    def __del__(self):
        getattr(self, "_static_tensor_cache", {}).clear()
        getattr(self, "_weather_tensor_cache", {}).clear()
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.close()
