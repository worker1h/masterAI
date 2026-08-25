"""Calibrate and promote the best GFF checkpoint on the validation split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import DataLoader

from src.gff_data import WEATHER_CHANNELS
from src.gff_model import build_gff_model
from src.train_gff import (
    evaluate,
    make_dataset,
    seed_everything,
    select_validation_adaptive_gate,
    select_validation_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or output_dir / "best.pt"
    postprocessing = dict(config.get("postprocessing", {}))
    probability_normalization = str(
        postprocessing.get("probability_normalization", "none")
    )
    adaptive = probability_normalization == "adaptive_low_confidence_minmax"
    workers = int(config.get("num_workers", 0))
    loader_options = {
        "batch_size": int(config["batch_size"]),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    validation_dataset = make_dataset(config, "val")
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_gff_model(
        config, validation_dataset.spatial_channels, WEATHER_CHANNELS
    ).to(device)
    candidates = [float(value) for value in config["threshold_candidates"]]
    ranking: list[dict] = []
    checkpoints: dict[str, dict] = {}
    started = time.time()
    for value in args.checkpoints:
        path = Path(value)
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if adaptive:
            gate, scores, fractions = select_validation_adaptive_gate(
                model,
                validation_loader,
                device,
                postprocessing,
                [float(value) for value in postprocessing["confidence_gate_candidates"]],
            )
            item = {
                "checkpoint": str(path),
                "epoch": int(state["epoch"]),
                "selected_confidence_gate": gate,
                "raw_threshold": float(postprocessing.get("raw_threshold", 0.3)),
                "normalized_threshold": float(
                    postprocessing.get("normalized_threshold", 0.75)
                ),
                "validation_macro_iou": float(scores[str(gate)]),
                "confidence_gate_scores": scores,
                "normalized_fractions": fractions,
            }
        else:
            threshold, scores = select_validation_threshold(
                model,
                validation_loader,
                device,
                candidates,
                probability_normalization,
            )
            item = {
                "checkpoint": str(path),
                "epoch": int(state["epoch"]),
                "selected_threshold": threshold,
                "validation_macro_iou": float(scores[str(threshold)]),
                "threshold_scores": scores,
            }
        ranking.append(item)
        checkpoints[str(path)] = state
        print(json.dumps(item, ensure_ascii=False))
    ranking.sort(key=lambda item: item["validation_macro_iou"], reverse=True)
    winner = ranking[0]
    state = checkpoints[winner["checkpoint"]]
    model.load_state_dict(state["model"])
    effective_postprocessing = dict(postprocessing)
    if adaptive:
        effective_postprocessing["confidence_gate"] = float(
            winner["selected_confidence_gate"]
        )
        threshold = float(effective_postprocessing.get("raw_threshold", 0.3))
    else:
        threshold = float(winner["selected_threshold"])
    validation_loss, validation_metrics = evaluate(
        model,
        validation_loader,
        device,
        config.get("loss", {}),
        threshold,
        probability_normalization,
        effective_postprocessing,
    )
    test_dataset = make_dataset(config, "test")
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        device,
        config.get("loss", {}),
        threshold,
        probability_normalization,
        effective_postprocessing,
    )
    state["selected_threshold"] = threshold
    if adaptive:
        state["selected_normalized_threshold"] = float(
            effective_postprocessing["normalized_threshold"]
        )
        state["selected_confidence_gate"] = float(
            effective_postprocessing["confidence_gate"]
        )
        state["confidence_gate_scores"] = winner["confidence_gate_scores"]
    else:
        state["threshold_scores"] = winner["threshold_scores"]
    state["validation_metrics"] = validation_metrics
    state["test_metrics"] = test_metrics
    torch.save(state, output_path)

    prior_path = output_dir / "run.json"
    prior = (
        json.loads(prior_path.read_text(encoding="utf-8"))
        if prior_path.exists()
        else {}
    )
    summary = {
        "experiment_type": "causal hindcast (no post-issue observations)",
        "operational_forecast": False,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "samples": {
            "train": len(make_dataset(config, "train")),
            "val": len(validation_dataset),
            "test": len(test_dataset),
        },
        "best_epoch": int(state["epoch"]),
        "checkpoint_validation_at_training_threshold": state.get("metrics", {}),
        "selected_threshold": threshold,
        "selected_normalized_threshold": (
            float(effective_postprocessing["normalized_threshold"])
            if adaptive
            else None
        ),
        "selected_confidence_gate": (
            float(effective_postprocessing["confidence_gate"])
            if adaptive
            else None
        ),
        "probability_normalization": probability_normalization,
        "postprocessing": effective_postprocessing,
        "threshold_validation_macro_iou": winner.get("threshold_scores", {}),
        "confidence_gate_validation_macro_iou": winner.get(
            "confidence_gate_scores", {}
        ),
        "confidence_gate_normalized_fractions": winner.get(
            "normalized_fractions", {}
        ),
        "checkpoint_ranking": ranking,
        "validation_loss": validation_loss,
        "validation": validation_metrics,
        "test_loss": test_loss,
        "test": test_metrics,
        "elapsed_seconds": float(prior.get("elapsed_seconds", 0.0))
        + (time.time() - started),
        "config": config,
    }
    prior_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "checkpoint_ranking.json").write_text(
        json.dumps(ranking, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "test_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        row = {"test_loss": test_loss, **test_metrics}
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
