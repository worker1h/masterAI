import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
import zarr
from zarr.storage import ZipStore


SAMPLE_PATTERN = re.compile(
    r"^(?P<event>EMSR\d+)_(?P<area>\d+)_(?P<grid>[^_]+)_x(?P<x>\d+)_y(?P<y>\d+)$"
)
TILE_PIXELS = 256
RESOLUTION = 10


def read_event_time(path):
    with ZipStore(path, mode="r") as store:
        group = zarr.open_group(store, mode="r")
        times = np.asarray(group["time"][:]).astype("datetime64[ns]").astype(str)
    return times[2]


def partition(time_text):
    year = int(time_text[:4])
    if year <= 2023:
        return "train"
    if year == 2024:
        return "val"
    return "test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output", default="data/split/future_event_forecast_manifest.json")
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    s1_index = {}
    for path in data_root.rglob("*_S1RTC.zarr.zip"):
        split = path.relative_to(data_root).parts[0]
        name = path.name.removesuffix("_S1RTC.zarr.zip")
        s1_index[(split, name)] = path
    records = []
    for mask_path in data_root.rglob("*_annotation_flood.tif"):
        name = mask_path.name.removesuffix("_annotation_flood.tif")
        match = SAMPLE_PATTERN.match(name)
        if match is None:
            continue
        split = mask_path.relative_to(data_root).parts[0]
        s1_path = s1_index.get((split, name))
        if s1_path is None:
            continue
        records.append(
            {
                "name": name,
                "split": split,
                "mask_path": mask_path.as_posix(),
                "s1_path": s1_path.as_posix(),
                **match.groupdict(),
            }
        )
    event_times = {}
    for item in records:
        if item["event"] not in event_times:
            event_times[item["event"]] = read_event_time(item["s1_path"])
    by_grid = defaultdict(lambda: defaultdict(list))
    for item in records:
        by_grid[item["grid"]][item["event"]].append(item)
    samples = []
    for grid, events in by_grid.items():
        ordered_events = sorted(events, key=event_times.get)
        for history_event, future_event in zip(ordered_events, ordered_events[1:]):
            for future in events[future_event]:
                coverage = np.zeros((TILE_PIXELS, TILE_PIXELS), dtype=bool)
                history_tiles = []
                future_x = int(future["x"])
                future_y = int(future["y"])
                for history in events[history_event]:
                    column_offset = round((int(history["x"]) - future_x) / RESOLUTION)
                    row_offset = round((future_y - int(history["y"])) / RESOLUTION)
                    row_start = max(0, row_offset)
                    column_start = max(0, column_offset)
                    row_end = min(TILE_PIXELS, row_offset + TILE_PIXELS)
                    column_end = min(TILE_PIXELS, column_offset + TILE_PIXELS)
                    if row_end <= row_start or column_end <= column_start:
                        continue
                    coverage[row_start:row_end, column_start:column_end] = True
                    history_tiles.append(
                        {
                            "name": history["name"],
                            "s1_path": history["s1_path"],
                            "row_offset": row_offset,
                            "column_offset": column_offset,
                        }
                    )
                coverage_fraction = float(coverage.mean())
                if coverage_fraction < args.minimum_coverage:
                    continue
                with rasterio.open(future["mask_path"]) as source:
                    target = source.read(1)
                samples.append(
                    {
                        "partition": partition(event_times[future_event]),
                        "grid": grid,
                        "history_event": history_event,
                        "history_event_time": event_times[history_event],
                        "future_event": future_event,
                        "future_event_time": event_times[future_event],
                        "future_name": future["name"],
                        "future_mask_path": future["mask_path"],
                        "future_flood_pixels": int((np.nan_to_num(target) == 2).sum()),
                        "coverage_fraction": coverage_fraction,
                        "history_tiles": history_tiles,
                    }
                )
    samples.sort(key=lambda item: (item["future_event_time"], item["future_name"]))
    counts = Counter(sample["partition"] for sample in samples)
    manifest = {
        "task": "forecast the next flood-event mask from the previous event's pre/event SAR",
        "minimum_history_coverage": args.minimum_coverage,
        "temporal_partition": {"train": "<=2023", "val": "2024", "test": ">=2025"},
        "counts": dict(counts),
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
