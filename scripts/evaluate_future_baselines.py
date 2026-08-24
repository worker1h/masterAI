import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FutureEventForecastDataset
from src.metrics import binary_metrics
from src.model import build_model


def counts(prediction, target):
    prediction = prediction.bool()
    target = target.bool()
    return np.asarray(
        [
            torch.logical_and(prediction, target).sum().item(),
            torch.logical_and(prediction, ~target).sum().item(),
            torch.logical_and(~prediction, target).sum().item(),
            torch.logical_and(~prediction, ~target).sum().item(),
        ],
        dtype=np.float64,
    )


def summarize(name, total, per_event):
    event_ious = [binary_metrics(*value)["iou"] for value in per_event.values()]
    return {
        "name": name,
        **binary_metrics(*total),
        "event_macro_iou": float(np.mean(event_ious)),
        "events": len(per_event),
    }


def historical_mask_mosaic(sample, mask_index):
    mosaic = np.zeros((256, 256), dtype=bool)
    for tile in sample["history_tiles"]:
        with rasterio.open(mask_index[tile["name"]]) as source:
            mask = source.read(1) == 2
        row_offset = int(tile["row_offset"])
        column_offset = int(tile["column_offset"])
        target_row_start = max(0, row_offset)
        target_column_start = max(0, column_offset)
        target_row_end = min(256, row_offset + 256)
        target_column_end = min(256, column_offset + 256)
        if target_row_end <= target_row_start or target_column_end <= target_column_start:
            continue
        source_row_start = target_row_start - row_offset
        source_column_start = target_column_start - column_offset
        source_row_end = source_row_start + target_row_end - target_row_start
        source_column_end = source_column_start + target_column_end - target_column_start
        mosaic[target_row_start:target_row_end, target_column_start:target_column_end] |= mask[
            source_row_start:source_row_end, source_column_start:source_column_end
        ]
    return torch.from_numpy(mosaic)[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-config", required=True)
    parser.add_argument("--mapping-config", required=True)
    parser.add_argument("--partition", default="test")
    args = parser.parse_args()
    forecast_config = yaml.safe_load(Path(args.forecast_config).read_text(encoding="utf-8"))
    mapping_config = yaml.safe_load(Path(args.mapping_config).read_text(encoding="utf-8"))
    dataset = FutureEventForecastDataset(forecast_config["manifest"], args.partition)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping_model = build_model(
        mapping_config["model"], 4, mapping_config["base_channels"]
    ).to(device)
    state = torch.load(
        Path(mapping_config["output_dir"]) / "best.pt", map_location=device, weights_only=False
    )
    mapping_model.load_state_dict(state["model"])
    mapping_model.eval()
    data_root = Path(forecast_config["data_root"])
    mask_index = {
        path.name.removesuffix("_annotation_flood.tif"): path
        for path in data_root.rglob("*_annotation_flood.tif")
    }
    totals = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    per_event = defaultdict(lambda: defaultdict(lambda: np.zeros(4, dtype=np.float64)))
    with torch.inference_mode():
        for index, (inputs, target, names) in enumerate(loader):
            target = target.bool()
            event = names[0].split("_", 1)[0]
            predictions = {
                "all_empty": torch.zeros_like(target),
                "all_flood": torch.ones_like(target),
                "oracle_previous_mask": historical_mask_mosaic(
                    dataset.samples[index], mask_index
                )[None],
            }
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                historical_mapping = torch.sigmoid(mapping_model(inputs.to(device))) >= 0.5
            predictions["mapped_previous_flood"] = historical_mapping.cpu()
            for name, prediction in predictions.items():
                value = counts(prediction, target)
                totals[name] += value
                per_event[name][event] += value
    results = [summarize(name, total, per_event[name]) for name, total in totals.items()]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
