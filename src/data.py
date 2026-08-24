import json
from pathlib import Path
import numpy as np
import torch
import rasterio
import zarr
from zarr.storage import ZipStore
from torch.utils.data import Dataset


CHANNELS = {"e0": 2, "e1": 3, "e2": 4, "e3": 5}
S1_MEAN = np.asarray([-9.98, -15.968], dtype=np.float32)[:, None, None]
S1_STD = np.asarray([4.24, 4.105], dtype=np.float32)[:, None, None]
DEM_MEAN = 141.786
DEM_STD = 189.363


def _read_zarr(path: Path, key="bands"):
    with ZipStore(path, mode="r") as store:
        group = zarr.open_group(store, mode="r")
        return np.asarray(group[key][...])


def _to_chw(a):
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 2:
        a = a[None]
    if a.shape[-1] <= 8 and a.shape[0] > 8:
        a = np.moveaxis(a, -1, 0)
    return a


def _normalize_s1(a):
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    repeats = a.shape[0] // 2
    return (a - np.tile(S1_MEAN, (repeats, 1, 1))) / np.tile(S1_STD, (repeats, 1, 1))


class ImpactMeshDataset(Dataset):
    def __init__(self, root, split, mode="e0", limit=None, sample_names=None):
        self.root = Path(root) / split
        self.mode = mode
        masks = sorted((self.root / "MASK").rglob("*_annotation_flood.tif"))
        self.mask_paths = {p.name.removesuffix("_annotation_flood.tif"): p for p in masks}
        self.s1_paths = {p.name.removesuffix("_S1RTC.zarr.zip"): p for p in (self.root / "S1RTC").rglob("*_S1RTC.zarr.zip")}
        self.dem_paths = {p.name.removesuffix("_DEM.tif"): p for p in (self.root / "DEM").rglob("*_DEM.tif")}
        need_dem = mode in ("e1", "e3")
        allowed = set(sample_names) if sample_names is not None else None
        self.names = sorted(name for name in self.mask_paths if name in self.s1_paths and (not need_dem or name in self.dem_paths) and (allowed is None or name in allowed))
        if limit is not None:
            self.names = self.names[: int(limit)]
        if not self.names:
            raise FileNotFoundError(f"No aligned samples found below {self.root}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        raw = np.asarray(_read_zarr(self.s1_paths[name]), dtype=np.float32)
        if raw.ndim == 4:  # time, band, y, x
            s1 = raw.reshape(raw.shape[0] * raw.shape[1], raw.shape[2], raw.shape[3])
        else:
            s1 = _to_chw(raw)
        # Official S1 arrays contain four time observations; each has VV/VH.
        if s1.shape[0] >= 8:
            pre, event = s1[2:4], s1[4:6]
        elif s1.shape[0] >= 4:
            pre, event = s1[0:2], s1[-2:]
        else:
            pre = event = s1[:2]
        parts = [event]
        if self.mode in ("e2", "e3"):
            parts = [pre, event]
        if self.mode in ("e1", "e3"):
            with rasterio.open(self.dem_paths[name]) as src:
                dem = src.read(1).astype(np.float32)[None]
            dem = (np.nan_to_num(dem, nan=DEM_MEAN) - DEM_MEAN) / DEM_STD
            parts.append(dem)
        sar_count = 4 if self.mode in ("e2", "e3") else 2
        x = np.concatenate(parts, axis=0).astype(np.float32)
        x[:sar_count] = _normalize_s1(x[:sar_count])
        with rasterio.open(self.mask_paths[name]) as src:
            mask = src.read(1).astype(np.float32)[None]
        # ImpactMesh-Flood: 0 background, 1 permanent water, 2 flood.
        # Only class 2 is the transient flood target; permanent water is negative.
        mask = (np.nan_to_num(mask) == 2).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(mask), name


class FutureEventForecastDataset(Dataset):
    """Historical pre/event SAR mosaics paired with the next flood-event mask."""

    def __init__(self, manifest_path, partition):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.samples = [sample for sample in manifest["samples"] if sample["partition"] == partition]
        if not self.samples:
            raise FileNotFoundError(
                f"No future-event forecast samples for partition {partition!r} in {manifest_path}"
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _historical_sar(path):
        raw = np.asarray(_read_zarr(Path(path)), dtype=np.float32)
        if raw.ndim == 4:
            sar = raw.reshape(raw.shape[0] * raw.shape[1], raw.shape[2], raw.shape[3])
        else:
            sar = _to_chw(raw)
        if sar.shape[0] < 6:
            raise ValueError(f"Expected four S1 time observations in {path}")
        return np.concatenate([sar[2:4], sar[4:6]], axis=0)

    def __getitem__(self, index):
        sample = self.samples[index]
        height = width = 256
        channels = 4
        accumulated = np.zeros((channels, height, width), dtype=np.float32)
        counts = np.zeros((1, height, width), dtype=np.float32)
        channel_means = np.tile(S1_MEAN[:, 0, 0], 2)[:, None, None]
        for tile in sample["history_tiles"]:
            sar = self._historical_sar(tile["s1_path"])
            sar = np.where(np.isfinite(sar), sar, channel_means)
            row_offset = int(tile["row_offset"])
            column_offset = int(tile["column_offset"])
            target_row_start = max(0, row_offset)
            target_column_start = max(0, column_offset)
            target_row_end = min(height, row_offset + sar.shape[1])
            target_column_end = min(width, column_offset + sar.shape[2])
            if target_row_end <= target_row_start or target_column_end <= target_column_start:
                continue
            source_row_start = target_row_start - row_offset
            source_column_start = target_column_start - column_offset
            source_row_end = source_row_start + target_row_end - target_row_start
            source_column_end = source_column_start + target_column_end - target_column_start
            target_slice = (
                slice(None),
                slice(target_row_start, target_row_end),
                slice(target_column_start, target_column_end),
            )
            source_slice = (
                slice(None),
                slice(source_row_start, source_row_end),
                slice(source_column_start, source_column_end),
            )
            accumulated[target_slice] += sar[source_slice]
            counts[
                :,
                target_row_start:target_row_end,
                target_column_start:target_column_end,
            ] += 1
        raw_mosaic = np.broadcast_to(channel_means, accumulated.shape).copy()
        np.divide(accumulated, counts, out=raw_mosaic, where=counts > 0)
        inputs = _normalize_s1(raw_mosaic).astype(np.float32)
        with rasterio.open(sample["future_mask_path"]) as source:
            mask = source.read(1).astype(np.float32)[None]
        target = (np.nan_to_num(mask) == 2).astype(np.float32)
        return torch.from_numpy(inputs), torch.from_numpy(target), sample["future_name"]
