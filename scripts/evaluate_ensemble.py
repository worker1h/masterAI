import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_checkpoint import boundary_counts
from src.data import CHANNELS, ImpactMeshDataset
from src.metrics import binary_metrics
from src.model import build_model


def load_models(config_paths, device):
    configs = [yaml.safe_load(Path(path).read_text(encoding="utf-8")) for path in config_paths]
    modes = {cfg["input_mode"] for cfg in configs}
    if len(modes) != 1:
        raise ValueError("All ensemble members must use the same input mode")
    models = []
    for cfg in configs:
        state = torch.load(
            Path(cfg["output_dir"]) / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model = build_model(
            cfg.get("model", "unet"),
            CHANNELS[cfg["input_mode"]],
            cfg["base_channels"],
        ).to(device)
        model.load_state_dict(state["model"])
        models.append(model.eval())
    return configs, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-list")
    parser.add_argument("--name", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    weights = args.weights or [1.0] * len(args.configs)
    if (
        len(weights) != len(args.configs)
        or any(weight < 0 for weight in weights)
        or sum(weights) <= 0
    ):
        raise ValueError("Provide one non-negative weight per config")
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs, models = load_models(args.configs, device)
    names = None
    if args.sample_list:
        names = [
            value.strip()
            for value in Path(args.sample_list).read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
    cfg = configs[0]
    dataset = ImpactMeshDataset(
        cfg["data_root"], args.split, cfg["input_mode"], sample_names=names
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    counts = defaultdict(lambda: np.zeros(4))
    boundary = np.zeros(4, dtype=np.float64)
    tensor_weights = torch.as_tensor(weights, device=device, dtype=torch.float32)
    with torch.no_grad():
        for inputs, targets, samples in loader:
            inputs = inputs.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                probabilities = torch.stack(
                    [torch.sigmoid(model(inputs)) for model in models], dim=0
                )
                probabilities = (
                    probabilities * tensor_weights[:, None, None, None, None]
                ).sum(dim=0)
            predictions = probabilities >= 0.5
            truths = targets.to(device) >= 0.5
            boundary += boundary_counts(predictions, truths)
            for index, sample in enumerate(samples):
                event = sample.split("_", 1)[0]
                pred = predictions[index]
                truth = truths[index]
                counts[event] += [
                    torch.logical_and(pred, truth).sum().item(),
                    torch.logical_and(pred, ~truth).sum().item(),
                    torch.logical_and(~pred, truth).sum().item(),
                    torch.logical_and(~pred, ~truth).sum().item(),
                ]

    total = sum(counts.values(), np.zeros(4))
    boundary_precision = boundary[0] / max(boundary[1], 1)
    boundary_recall = boundary[2] / max(boundary[3], 1)
    boundary_f1 = 2 * boundary_precision * boundary_recall / max(
        boundary_precision + boundary_recall, 1e-12
    )
    summary = {
        "name": args.name,
        "samples": len(dataset),
        "events": len(counts),
        "members": [cfg.get("model", "unet") for cfg in configs],
        "weights": weights.tolist(),
        **binary_metrics(*total),
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "boundary_tolerance_pixels": 2,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.name}_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = [
        {
            "event": event,
            "truth_positive_pixels": int(values[0] + values[2]),
            "predicted_positive_pixels": int(values[0] + values[1]),
            **binary_metrics(*values),
        }
        for event, values in sorted(counts.items())
    ]
    with (output_dir / f"{args.name}_per_event.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
