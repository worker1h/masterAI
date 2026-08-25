import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FutureEventForecastDataset
from src.metrics import binary_metrics
from src.model import build_model


def collect(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    targets = []
    with torch.inference_mode():
        for inputs, target, _ in loader:
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                probability = torch.sigmoid(model(inputs.to(device)))
            probabilities.append(probability.cpu())
            targets.append(target.bool())
    return torch.cat(probabilities), torch.cat(targets)


def metrics_at(probabilities, targets, threshold):
    prediction = probabilities >= threshold
    tp = torch.logical_and(prediction, targets).sum().item()
    fp = torch.logical_and(prediction, ~targets).sum().item()
    fn = torch.logical_and(~prediction, targets).sum().item()
    tn = torch.logical_and(~prediction, ~targets).sum().item()
    return binary_metrics(tp, fp, fn, tn)


def evaluate(config_path, manifest, device):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model = build_model(
        config["model"], int(config.get("input_channels", 4)), config["base_channels"]
    ).to(device)
    state = torch.load(
        Path(config["output_dir"]) / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(state["model"])
    model.eval()
    validation = FutureEventForecastDataset(manifest, "val")
    test = FutureEventForecastDataset(manifest, "test")
    val_probability, val_target = collect(model, validation, device, config["batch_size"])
    thresholds = np.arange(0.05, 0.951, 0.05)
    candidates = [
        (float(threshold), metrics_at(val_probability, val_target, float(threshold)))
        for threshold in thresholds
    ]
    threshold, validation_metrics = max(candidates, key=lambda item: item[1]["iou"])
    test_probability, test_target = collect(model, test, device, config["batch_size"])
    return {
        "config": str(config_path),
        "model": config["model"],
        "selected_threshold": threshold,
        "validation": validation_metrics,
        "test": metrics_at(test_probability, test_target, threshold),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [evaluate(path, args.manifest, device) for path in args.configs]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
