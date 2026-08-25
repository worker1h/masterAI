import argparse
import json
import statistics
import time
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CHANNELS
from src.model import build_model


def benchmark(config_path, batch_size, warmup, repeats):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config.get("model", "unet"),
        CHANNELS[config["input_mode"]],
        config["base_channels"],
    ).to(device)
    checkpoint = torch.load(
        Path(config["output_dir"]) / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    inputs = torch.randn(batch_size, CHANNELS[config["input_mode"]], 256, 256, device=device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                model(inputs)
        synchronize()
        elapsed = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                model(inputs)
            synchronize()
            elapsed.append((time.perf_counter() - start) * 1000 / batch_size)

    return {
        "config": str(config_path),
        "model": config.get("model", "unet"),
        "device": str(device),
        "batch_size": batch_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "median_latency_ms_per_image": statistics.median(elapsed),
        "mean_latency_ms_per_image": statistics.mean(elapsed),
        "warmup": warmup,
        "repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=60)
    args = parser.parse_args()
    for config_path in args.configs:
        print(
            json.dumps(
                benchmark(config_path, args.batch_size, args.warmup, args.repeats),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
