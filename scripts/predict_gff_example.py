"""Render one GFF 24/48/72-hour forecast against the flooded-class target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.gff_data import GFFFloodForecastDataset, WEATHER_CHANNELS
from src.gff_model import build_gff_model
from src.train_gff import flood_probability, postprocess_probability_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--horizon", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--index", type=int)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Defaults to the validation-selected threshold stored in the checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/gff_prediction_example.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = config["data"]
    dataset = GFFFloodForecastDataset(
        root=data["root"],
        split=args.split,
        fold=int(data.get("fold", 0)),
        horizons=tuple(data.get("horizons", [1, 2, 3])),
        weather_window=int(data.get("weather_window", 20)),
        tile_size=int(data.get("tile_size", 224)),
        context_size=int(data.get("context_size", 16)),
        context_buffer_m=float(data.get("context_buffer_m", 50_000.0)),
        max_tiles=data.get(f"max_{args.split}_tiles"),
        max_sites=data.get(f"max_{args.split}_sites"),
        seed=int(config["seed"]),
        augment=False,
        forcing_mode=str(data.get("forcing_mode", "causal")),
        sar_preprocessing=str(data.get("sar_preprocessing", "standard")),
        sar_db_ranges=tuple(
            tuple(float(item) for item in pair)
            for pair in data.get("sar_db_ranges", [[-25.0, 0.0], [-32.0, -5.0]])
        ),
        clahe_clip_limit=float(data.get("clahe_clip_limit", 2.0)),
        clahe_grid_size=int(data.get("clahe_grid_size", 8)),
        clahe_enhancement_size=int(data.get("clahe_enhancement_size", 256)),
    )
    if args.horizon not in dataset.horizons:
        raise ValueError(f"Horizon {args.horizon} is absent from configured horizons")
    horizon_offset = dataset.horizons.index(args.horizon)
    if args.index is None:
        positive_tile = next(
            (index for index, value in enumerate(dataset.tile_positive_flags) if value), 0
        )
        index = positive_tile * len(dataset.horizons) + horizon_offset
    else:
        index = int(args.index)
    sample = dataset[index]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    threshold = float(
        args.threshold
        if args.threshold is not None
        else checkpoint.get("selected_threshold", 0.5)
    )
    model = build_gff_model(config, dataset.spatial_channels, WEATHER_CHANNELS)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        outputs = model(
            sample["spatial"][None].to(device),
            sample["weather"][None].to(device),
            sample["horizon"][None].to(device),
            sample["forecast_mask"][None].to(device),
        )
    raw_probability = flood_probability(outputs)
    postprocessing = dict(config.get("postprocessing", {}))
    if "selected_confidence_gate" in checkpoint:
        postprocessing["confidence_gate"] = float(
            checkpoint["selected_confidence_gate"]
        )
    probability_normalization = str(
        postprocessing.get("probability_normalization", "none")
    )
    if args.threshold is not None and probability_normalization == (
        "adaptive_low_confidence_minmax"
    ):
        postprocessing["raw_threshold"] = float(args.threshold)
        postprocessing["normalized_threshold"] = float(args.threshold)
    processed = postprocess_probability_maps(
        raw_probability,
        sample["valid"][None].to(device),
        threshold,
        postprocessing,
    )
    probability = processed["probability"][0, 0].cpu().float().numpy()
    prediction = processed["prediction"][0, 0].cpu().numpy()
    was_normalized = bool(processed["normalized"][0].item())
    effective_threshold = float(processed["threshold"][0].item())
    confidence_value = float(processed["confidence"][0].item())
    confidence = confidence_value if np.isfinite(confidence_value) else None
    target = sample["target"][0].numpy() >= 0.5
    valid = sample["valid"][0].numpy() >= 0.5
    true_positive = prediction & target & valid
    false_positive = prediction & ~target & valid
    false_negative = ~prediction & target & valid
    errors = np.zeros((*target.shape, 3), dtype=np.float32)
    errors[true_positive] = (0.1, 0.8, 0.2)
    errors[false_positive] = (1.0, 0.2, 0.15)
    errors[false_negative] = (0.15, 0.35, 1.0)

    enhanced_sar = str(data.get("sar_preprocessing", "standard")) == "sunet_clahe"
    columns = 5 if enhanced_sar else 4
    figure, axes = plt.subplots(
        1, columns, figsize=(4 * columns, 4), constrained_layout=True
    )
    axes[0].imshow(
        sample["spatial"][0].numpy(),
        cmap="gray",
        vmin=-1 if enhanced_sar else None,
        vmax=1 if enhanced_sar else None,
        interpolation="nearest",
    )
    axes[0].set_title("Pre-event VV (dB)" if enhanced_sar else "Pre-event S1 VV")
    target_axis = 1
    if enhanced_sar:
        axes[1].imshow(
            sample["spatial"][2].numpy(),
            cmap="gray",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        axes[1].set_title("SU-Net CLAHE VV")
        target_axis = 2
    axes[target_axis].imshow(target, cmap="Blues", vmin=0, vmax=1)
    axes[target_axis].set_title("Observed flooded class")
    probability_axis = target_axis + 1
    image = axes[probability_axis].imshow(probability, cmap="magma", vmin=0, vmax=1)
    axes[probability_axis].contour(
        prediction, levels=[0.5], colors="cyan", linewidths=0.6
    )
    if probability_normalization == "adaptive_low_confidence_minmax":
        probability_title = (
            "Normalized probability (low confidence)"
            if was_normalized
            else "Raw forecast probability (high confidence)"
        )
    elif probability_normalization == "per_heatmap_minmax":
        probability_title = "Normalized probability"
    else:
        probability_title = "Forecast probability"
    axes[probability_axis].set_title(
        f"{probability_title} ({args.horizon * 24} h)"
    )
    figure.colorbar(image, ax=axes[probability_axis], fraction=0.046)
    axes[probability_axis + 1].imshow(errors, interpolation="nearest")
    axes[probability_axis + 1].set_title("TP green / FP red / FN blue")
    for axis in axes:
        axis.axis("off")
    forcing_mode = str(data.get("forcing_mode", "causal"))
    experiment_type = (
        "causal hindcast" if forcing_mode == "causal" else "perfect-forcing hindcast"
    )
    figure.suptitle(f"GFF site {sample['site']} | {experiment_type}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    fn = int(false_negative.sum())
    metadata = {
        "site": sample["site"],
        "dataset_index": index,
        "horizon_hours": args.horizon * 24,
        "threshold": effective_threshold,
        "probability_normalization": probability_normalization,
        "postprocessing_branch": "normalized" if was_normalized else "raw",
        "heatmap_confidence": confidence,
        "confidence_gate": postprocessing.get("confidence_gate"),
        "confidence_quantile": postprocessing.get("confidence_quantile"),
        "raw_probability_min": float(raw_probability.min().cpu()),
        "raw_probability_max": float(raw_probability.max().cpu()),
        "iou": tp / max(tp + fp + fn, 1),
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "experiment_type": experiment_type,
        "figure": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
