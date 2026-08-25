import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
import zarr
from zarr.storage import ZipStore


SAMPLE_PATTERN = re.compile(
    r"^(?P<event>EMSR\d+)_(?P<area>\d+)_(?P<grid>[^_]+)_(?P<x>x\d+)_(?P<y>y\d+)$"
)


def read_times(path):
    with ZipStore(path, mode="r") as store:
        group = zarr.open_group(store, mode="r")
        return np.asarray(group["time"][:]).astype("datetime64[ns]").astype(str).tolist()


def flood_pixels(path):
    with rasterio.open(path) as source:
        mask = source.read(1)
    return int((np.nan_to_num(mask) == 2).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/future_prediction_audit")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    s1_index = {}
    for s1_path in data_root.rglob("*_S1RTC.zarr.zip"):
        split = s1_path.relative_to(data_root).parts[0]
        name = s1_path.name.removesuffix("_S1RTC.zarr.zip")
        s1_index[(split, name)] = s1_path
    records = []
    by_spatial_key = defaultdict(list)
    for mask_path in data_root.rglob("*_annotation_flood.tif"):
        name = mask_path.name.removesuffix("_annotation_flood.tif")
        match = SAMPLE_PATTERN.match(name)
        if match is None:
            continue
        split = mask_path.relative_to(data_root).parts[0]
        s1_path = s1_index.get((split, name))
        if s1_path is None:
            continue
        item = {
            "name": name,
            "split": split,
            "mask_path": str(mask_path),
            "s1_path": str(s1_path),
            **match.groupdict(),
        }
        item["spatial_key"] = "_".join((item["grid"], item["x"], item["y"]))
        records.append(item)
        by_spatial_key[item["spatial_key"]].append(item)

    repeated = {
        key: items
        for key, items in by_spatial_key.items()
        if len({item["event"] for item in items}) > 1
    }
    timestamp_cache = {}
    event_time = {}
    for item in records:
        if item["event"] in event_time:
            continue
        times = read_times(item["s1_path"])
        timestamp_cache[item["name"]] = times
        event_time[item["event"]] = times[2]
    pairs = []
    for spatial_key, items in repeated.items():
        enriched = []
        for item in items:
            times = timestamp_cache.setdefault(item["name"], read_times(item["s1_path"]))
            enriched.append(
                {
                    **item,
                    "pre_month_time": times[0],
                    "pre_event_time": times[1],
                    "event_time": times[2],
                    "post_event_time": times[3],
                    "flood_pixels": flood_pixels(item["mask_path"]),
                }
            )
        enriched.sort(key=lambda item: item["event_time"])
        for history, future in zip(enriched, enriched[1:]):
            if history["event"] == future["event"]:
                continue
            delta_days = (
                np.datetime64(future["event_time"]) - np.datetime64(history["event_time"])
            ) / np.timedelta64(1, "D")
            pairs.append(
                {
                    "spatial_key": spatial_key,
                    "history_event": history["event"],
                    "history_sample": history["name"],
                    "history_event_time": history["event_time"],
                    "history_flood_pixels": history["flood_pixels"],
                    "future_event": future["event"],
                    "future_sample": future["name"],
                    "future_event_time": future["event_time"],
                    "future_flood_pixels": future["flood_pixels"],
                    "delta_days": float(delta_days),
                }
            )

    by_grid = defaultdict(list)
    for item in records:
        item["x_coordinate"] = int(item["x"][1:])
        item["y_coordinate"] = int(item["y"][1:])
        by_grid[item["grid"]].append(item)
    approximate_transitions = []
    tile_size = 2560
    for grid, items in by_grid.items():
        by_event = defaultdict(list)
        for item in items:
            by_event[item["event"]].append(item)
        ordered_events = sorted(by_event, key=lambda event: event_time[event])
        for history_event, future_event in zip(ordered_events, ordered_events[1:]):
            future_tiles_with_half_overlap = 0
            future_tiles_with_near_full_overlap = 0
            future_tiles_with_near_full_mosaic_coverage = 0
            maximum_overlap_fractions = []
            mosaic_coverage_fractions = []
            for future in by_event[future_event]:
                best = 0.0
                covered = 0.0
                for history in by_event[history_event]:
                    overlap_x = max(
                        0, tile_size - abs(future["x_coordinate"] - history["x_coordinate"])
                    )
                    overlap_y = max(
                        0, tile_size - abs(future["y_coordinate"] - history["y_coordinate"])
                    )
                    overlap_fraction = overlap_x * overlap_y / tile_size**2
                    best = max(best, overlap_fraction)
                    covered += overlap_fraction
                covered = min(covered, 1.0)
                maximum_overlap_fractions.append(best)
                mosaic_coverage_fractions.append(covered)
                future_tiles_with_half_overlap += best >= 0.5
                future_tiles_with_near_full_overlap += best >= 0.95
                future_tiles_with_near_full_mosaic_coverage += covered >= 0.95
            if future_tiles_with_half_overlap or future_tiles_with_near_full_mosaic_coverage:
                approximate_transitions.append(
                    {
                        "grid": grid,
                        "history_event": history_event,
                        "history_event_time": event_time[history_event],
                        "future_event": future_event,
                        "future_event_time": event_time[future_event],
                        "history_tiles": len(by_event[history_event]),
                        "future_tiles": len(by_event[future_event]),
                        "future_tiles_half_overlap": future_tiles_with_half_overlap,
                        "future_tiles_near_full_overlap": future_tiles_with_near_full_overlap,
                        "future_tiles_near_full_mosaic_coverage": future_tiles_with_near_full_mosaic_coverage,
                        "maximum_overlap_fraction": max(maximum_overlap_fractions),
                        "maximum_mosaic_coverage_fraction": max(mosaic_coverage_fractions),
                    }
                )

    pair_counts = Counter(
        f"{pair['history_event']}->{pair['future_event']}" for pair in pairs
    )
    summary = {
        "samples_with_mask_and_s1": len(records),
        "events": len({record["event"] for record in records}),
        "unique_spatial_keys": len(by_spatial_key),
        "repeated_exact_spatial_keys": len(repeated),
        "consecutive_cross_event_pairs": len(pairs),
        "distinct_event_transitions": len(pair_counts),
        "event_transition_counts": dict(pair_counts.most_common()),
        "future_positive_pairs": sum(pair["future_flood_pixels"] > 0 for pair in pairs),
        "future_empty_pairs": sum(pair["future_flood_pixels"] == 0 for pair in pairs),
        "approximate_consecutive_event_transitions": len(approximate_transitions),
        "future_tiles_with_at_least_half_single_tile_overlap": sum(
            item["future_tiles_half_overlap"] for item in approximate_transitions
        ),
        "future_tiles_with_near_full_single_tile_overlap": sum(
            item["future_tiles_near_full_overlap"] for item in approximate_transitions
        ),
        "future_tiles_with_near_full_mosaic_coverage": sum(
            item["future_tiles_near_full_mosaic_coverage"]
            for item in approximate_transitions
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if pairs:
        with (output_dir / "exact_spatial_pairs.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as output:
            writer = csv.DictWriter(output, fieldnames=pairs[0].keys())
            writer.writeheader()
            writer.writerows(pairs)
    if approximate_transitions:
        with (output_dir / "approximate_spatial_transitions.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as output:
            writer = csv.DictWriter(output, fieldnames=approximate_transitions[0].keys())
            writer.writeheader()
            writer.writerows(approximate_transitions)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
