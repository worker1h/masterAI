"""Rank GFF examples for qualitative inspection after quantitative evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.gff_data import WEATHER_CHANNELS
from src.gff_model import build_gff_model
from src.train_gff import flood_probability, make_dataset, postprocess_probability_maps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--horizon", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = make_dataset(config, args.split)
    horizon_offset = dataset.horizons.index(args.horizon)
    indexes = list(range(horizon_offset, len(dataset), len(dataset.horizons)))
    subset = Subset(dataset, indexes)
    workers = int(config.get("num_workers", 0))
    loader = DataLoader(
        subset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    threshold = float(checkpoint.get("selected_threshold", 0.5))
    postprocessing = dict(config.get("postprocessing", {}))
    if "selected_confidence_gate" in checkpoint:
        postprocessing["confidence_gate"] = float(
            checkpoint["selected_confidence_gate"]
        )
    probability_normalization = str(
        postprocessing.get("probability_normalization", "none")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_gff_model(config, dataset.spatial_channels, WEATHER_CHANNELS)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    rows: list[dict] = []
    offset = 0
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        for batch in loader:
            outputs = model(
                batch["spatial"].to(device),
                batch["weather"].to(device),
                batch["horizon"].to(device),
                batch["forecast_mask"].to(device),
            )
            processed = postprocess_probability_maps(
                flood_probability(outputs),
                batch["valid"].to(device),
                threshold,
                postprocessing,
            )
            prediction = processed["prediction"].cpu()
            normalized = processed["normalized"].cpu()
            confidence = processed["confidence"].cpu()
            thresholds = processed["threshold"].cpu()
            target = batch["target"] >= 0.5
            valid = batch["valid"] >= 0.5
            for local_index in range(prediction.shape[0]):
                predicted = prediction[local_index] & valid[local_index]
                observed = target[local_index] & valid[local_index]
                true_positive = int((predicted & observed).sum())
                false_positive = int((predicted & ~observed).sum())
                false_negative = int((~predicted & observed).sum())
                union = true_positive + false_positive + false_negative
                confidence_value = float(confidence[local_index].item())
                rows.append(
                    {
                        "dataset_index": indexes[offset + local_index],
                        "site": batch["site"][local_index],
                        "horizon_hours": args.horizon * 24,
                        "postprocessing_branch": (
                            "normalized"
                            if bool(normalized[local_index].item())
                            else "raw"
                        ),
                        "heatmap_confidence": (
                            confidence_value if math.isfinite(confidence_value) else None
                        ),
                        "threshold": float(thresholds[local_index].item()),
                        "iou": true_positive / max(union, 1),
                        "true_positive_pixels": true_positive,
                        "false_positive_pixels": false_positive,
                        "false_negative_pixels": false_negative,
                    }
                )
            offset += prediction.shape[0]

    positive_targets = [
        row
        for row in rows
        if row["true_positive_pixels"] + row["false_negative_pixels"] > 0
    ]
    positive_targets.sort(key=lambda row: row["iou"], reverse=True)
    result = {
        "selection_note": (
            "Post-evaluation qualitative ranking; never use this ranking for model, "
            "checkpoint, or threshold selection."
        ),
        "threshold": threshold,
        "probability_normalization": probability_normalization,
        "postprocessing": postprocessing,
        "evaluated_examples": len(rows),
        "positive_target_examples": len(positive_targets),
        "top_examples": positive_targets[: max(int(args.top_k), 1)],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
