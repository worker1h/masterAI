"""Train and evaluate GFFHorizonFormer on 24/48/72-hour hindcasts."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .gff_data import GFFFloodForecastDataset, WEATHER_CHANNELS
from .gff_model import build_gff_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def masked_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    probability = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def objective(outputs: dict[str, torch.Tensor], batch: dict, config: dict) -> torch.Tensor:
    target = batch["target"]
    valid = batch["valid"]
    pos_weight = torch.as_tensor(
        float(config.get("pos_weight", 4.0)),
        device=target.device,
        dtype=target.dtype,
    )
    bce = F.binary_cross_entropy_with_logits(
        outputs["segmentation"], target, pos_weight=pos_weight, reduction="none"
    )
    gamma = float(config.get("focal_gamma", 2.0))
    probability = torch.sigmoid(outputs["segmentation"])
    probability_true = probability * target + (1.0 - probability) * (1.0 - target)
    focal_bce = masked_mean((1.0 - probability_true).pow(gamma) * bce, valid)
    dice = masked_dice_loss(outputs["segmentation"], target, valid)

    boundary_weight = torch.as_tensor(
        float(config.get("boundary_pos_weight", 3.0)),
        device=target.device,
        dtype=target.dtype,
    )
    boundary_bce = F.binary_cross_entropy_with_logits(
        outputs["boundary"],
        batch["boundary"],
        pos_weight=boundary_weight,
        reduction="none",
    )
    boundary = masked_mean(boundary_bce, valid) + masked_dice_loss(
        outputs["boundary"], batch["boundary"], valid
    )
    presence = F.binary_cross_entropy_with_logits(outputs["presence"], batch["presence"])
    return (
        float(config.get("focal_weight", 1.0)) * focal_bce
        + float(config.get("dice_weight", 1.0)) * dice
        + float(config.get("boundary_weight", 0.2)) * boundary
        + float(config.get("presence_weight", 0.15)) * presence
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: (
            value.to(device, non_blocking=True)
            if torch.is_tensor(value) and key != "bounds"
            else value
        )
        for key, value in batch.items()
    }


def flood_probability(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Softly suppress pixel predictions when the tile-level head sees no flood."""

    presence = torch.sigmoid(outputs["presence"])[:, None, None, None]
    return torch.sigmoid(outputs["segmentation"]) * presence


def normalize_probability_maps(
    probability: torch.Tensor,
    valid: torch.Tensor | None = None,
    mode: str = "none",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize each heatmap independently over valid pixels before thresholding."""

    if mode == "none":
        return probability
    if mode != "per_heatmap_minmax":
        raise ValueError(f"Unsupported probability normalization: {mode}")
    mask = torch.ones_like(probability, dtype=torch.bool) if valid is None else valid.bool()
    minimum = torch.where(mask, probability, torch.inf).amin(
        dim=(-2, -1), keepdim=True
    )
    maximum = torch.where(mask, probability, -torch.inf).amax(
        dim=(-2, -1), keepdim=True
    )
    dynamic_range = maximum - minimum
    has_range = mask.any(dim=(-2, -1), keepdim=True) & (dynamic_range > eps)
    normalized = (probability - minimum) / dynamic_range.clamp_min(eps)
    return torch.where(has_range & mask, normalized.clamp(0.0, 1.0), 0.0)


def confusion_metrics(counts: np.ndarray) -> dict[str, float]:
    tp, fp, fn, tn = counts.astype(np.float64)
    eps = 1e-9
    return {
        "iou": float(tp / (tp + fp + fn + eps)),
        "dice": float(2.0 * tp / (2.0 * tp + fp + fn + eps)),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "accuracy": float((tp + tn) / (tp + fp + fn + tn + eps)),
    }


@torch.no_grad()
def select_validation_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    candidates: list[float],
    probability_normalization: str = "none",
) -> tuple[float, dict[str, float]]:
    """Choose one global threshold by validation macro IoU in a single pass."""

    model.eval()
    totals = {
        threshold: {horizon: np.zeros(4) for horizon in (1, 2, 3)}
        for threshold in candidates
    }
    for batch in tqdm(loader, desc="calibrate threshold", leave=False):
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            outputs = model(
                batch["spatial"],
                batch["weather"],
                batch["horizon"],
                batch["forecast_mask"],
            )
        probability = normalize_probability_maps(
            flood_probability(outputs), batch["valid"], probability_normalization
        )
        truth = batch["target"] >= 0.5
        valid = batch["valid"] >= 0.5
        for threshold in candidates:
            prediction = probability >= threshold
            for horizon in (1, 2, 3):
                selected = batch["horizon"] == horizon
                if not selected.any():
                    continue
                pred = prediction[selected] & valid[selected]
                target = truth[selected] & valid[selected]
                mask = valid[selected]
                totals[threshold][horizon] += np.array(
                    [
                        (pred & target).sum().item(),
                        (pred & ~target & mask).sum().item(),
                        (~pred & target & mask).sum().item(),
                        (~pred & ~target & mask).sum().item(),
                    ]
                )
    scores = {
        threshold: float(
            np.mean(
                [confusion_metrics(counts)["iou"] for counts in by_horizon.values()]
            )
        )
        for threshold, by_horizon in totals.items()
    }
    selected = max(scores, key=scores.get)
    return float(selected), {str(key): value for key, value in scores.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict,
    threshold: float = 0.5,
    probability_normalization: str = "none",
) -> tuple[float, dict[str, float]]:
    model.eval()
    totals = {1: np.zeros(4), 2: np.zeros(4), 3: np.zeros(4)}
    boundary_totals = {1: np.zeros(3), 2: np.zeros(3), 3: np.zeros(3)}
    loss_sum = 0.0
    examples = 0
    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            outputs = model(
                batch["spatial"],
                batch["weather"],
                batch["horizon"],
                batch["forecast_mask"],
            )
            loss = objective(outputs, batch, loss_config)
        loss_sum += float(loss) * batch["target"].shape[0]
        examples += batch["target"].shape[0]
        probability = normalize_probability_maps(
            flood_probability(outputs), batch["valid"], probability_normalization
        )
        boundary_probability = normalize_probability_maps(
            torch.sigmoid(outputs["boundary"]),
            batch["valid"],
            probability_normalization,
        )
        prediction = probability >= threshold
        boundary_prediction = boundary_probability >= threshold
        target = batch["target"] >= 0.5
        boundary_target = batch["boundary"] >= 0.5
        valid = batch["valid"] >= 0.5
        for horizon in totals:
            selected = batch["horizon"] == horizon
            if not selected.any():
                continue
            pred = prediction[selected] & valid[selected]
            truth = target[selected] & valid[selected]
            mask = valid[selected]
            totals[horizon] += np.array(
                [
                    (pred & truth).sum().item(),
                    (pred & ~truth & mask).sum().item(),
                    (~pred & truth & mask).sum().item(),
                    (~pred & ~truth & mask).sum().item(),
                ]
            )
            edge_pred = boundary_prediction[selected] & mask
            edge_truth = boundary_target[selected] & mask
            boundary_totals[horizon] += np.array(
                [
                    (edge_pred & edge_truth).sum().item(),
                    (edge_pred & ~edge_truth & mask).sum().item(),
                    (~edge_pred & edge_truth & mask).sum().item(),
                ]
            )
    metrics: dict[str, float] = {}
    horizon_ious = []
    for horizon in totals:
        values = confusion_metrics(totals[horizon])
        prefix = f"h{horizon * 24}"
        metrics.update({f"{prefix}_{name}": value for name, value in values.items()})
        edge_tp, edge_fp, edge_fn = boundary_totals[horizon]
        metrics[f"{prefix}_boundary_f1"] = float(
            2.0 * edge_tp / (2.0 * edge_tp + edge_fp + edge_fn + 1e-9)
        )
        horizon_ious.append(values["iou"])
    metrics["macro_iou"] = float(np.mean(horizon_ious))
    metrics["macro_dice"] = float(
        np.mean([metrics[f"h{horizon * 24}_dice"] for horizon in totals])
    )
    return loss_sum / max(examples, 1), metrics


def make_dataset(config: dict, split: str, strict: bool = True) -> GFFFloodForecastDataset:
    data = config["data"]
    return GFFFloodForecastDataset(
        root=data["root"],
        split=split,
        fold=int(data.get("fold", 0)),
        horizons=tuple(data.get("horizons", [1, 2, 3])),
        weather_window=int(data.get("weather_window", 20)),
        tile_size=int(data.get("tile_size", 224)),
        context_size=int(data.get("context_size", 16)),
        context_buffer_m=float(data.get("context_buffer_m", 50_000.0)),
        max_tiles=data.get(f"max_{split}_tiles"),
        max_sites=data.get(f"max_{split}_sites"),
        seed=int(config["seed"]),
        augment=split == "train" and bool(data.get("augment", True)),
        strict=strict,
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


class GroupedBalancedSampler(Sampler[int]):
    """Balance at tile level and emit all horizons together for cache locality."""

    def __init__(self, dataset: GFFFloodForecastDataset, weights: torch.Tensor, seed: int):
        self.dataset = dataset
        self.weights = weights
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        tile_indexes = torch.multinomial(
            self.weights,
            num_samples=len(self.dataset.tiles),
            replacement=True,
            generator=self.generator,
        )
        horizon_count = len(self.dataset.horizons)
        for tile_index in tile_indexes.tolist():
            order = torch.randperm(horizon_count, generator=self.generator).tolist()
            for horizon_index in order:
                yield tile_index * horizon_count + horizon_index

    def __len__(self) -> int:
        return len(self.dataset)


def balanced_sampler(dataset: GFFFloodForecastDataset, positive_ratio: float, seed: int):
    flags = np.asarray(dataset.tile_positive_flags, dtype=bool)
    positive_count = int(flags.sum())
    negative_count = int((~flags).sum())
    if not positive_count or not negative_count:
        return None
    weights = np.where(
        flags,
        positive_ratio / positive_count,
        (1.0 - positive_ratio) / negative_count,
    )
    return GroupedBalancedSampler(
        dataset,
        torch.as_tensor(weights, dtype=torch.double),
        seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        help="Resume model, optimizer, scheduler, and epoch from a training checkpoint.",
    )
    checkpoint_group.add_argument(
        "--init-checkpoint",
        help="Initialize model weights only; start a fresh optimizer and schedule.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    torch.set_float32_matmul_precision("high")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = make_dataset(config, "train")
    val_dataset = make_dataset(config, "val")
    sampler = balanced_sampler(
        train_dataset,
        float(config.get("sampling", {}).get("positive_tile_ratio", 0.5)),
        int(config["seed"]),
    )
    workers = int(config.get("num_workers", 0))
    loader_options = {
        "batch_size": int(config["batch_size"]),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_gff_model(
        config, train_dataset.spatial_channels, WEATHER_CHANNELS
    ).to(device)
    learning_rate = float(config["learning_rate"])
    backbone_learning_rate = config.get("backbone_learning_rate")
    if backbone_learning_rate is not None and hasattr(model, "vit"):
        backbone_parameters = [
            parameter for parameter in model.vit.parameters() if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        task_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        parameter_groups = [
            {"params": task_parameters, "lr": learning_rate},
            {"params": backbone_parameters, "lr": float(backbone_learning_rate)},
        ]
    else:
        parameter_groups = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    epochs = int(config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    best_iou = -1.0
    initialized_from = None
    if args.init_checkpoint:
        checkpoint = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint.get("model", checkpoint))
        initialized_from = str(Path(args.init_checkpoint))
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_iou = float(checkpoint.get("best_iou", -1.0))

    loss_config = config.get("loss", {})
    probability_normalization = str(
        config.get("postprocessing", {}).get("probability_normalization", "none")
    )
    accumulation = int(config.get("gradient_accumulation", 1))
    rows: list[dict] = []
    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        examples = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                outputs = model(
                    batch["spatial"],
                    batch["weather"],
                    batch["horizon"],
                    batch["forecast_mask"],
                )
                loss = objective(outputs, batch, loss_config)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_size = batch["target"].shape[0]
            running_loss += float(loss.detach()) * batch_size
            examples += batch_size
            progress.set_postfix(loss=f"{running_loss / examples:.4f}")
        scheduler.step()

        validation_loss, metrics = evaluate(
            model,
            val_loader,
            device,
            loss_config,
            float(config.get("threshold", 0.5)),
            probability_normalization,
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(examples, 1),
            "val_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_iou": max(best_iou, metrics["macro_iou"]),
            "metrics": metrics,
            "config": config,
            "initialized_from": initialized_from,
        }
        torch.save(state, output_dir / "last.pt")
        if bool(config.get("save_epoch_checkpoints", False)):
            torch.save(state, output_dir / f"epoch{epoch}.pt")
        if metrics["macro_iou"] > best_iou:
            best_iou = metrics["macro_iou"]
            torch.save(state, output_dir / "best.pt")

    if rows:
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    candidates = [
        float(value)
        for value in config.get(
            "threshold_candidates", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        )
    ]
    selected_threshold, threshold_scores = select_validation_threshold(
        model, val_loader, device, candidates, probability_normalization
    )
    calibrated_val_loss, calibrated_val_metrics = evaluate(
        model,
        val_loader,
        device,
        loss_config,
        selected_threshold,
        probability_normalization,
    )
    best_checkpoint["selected_threshold"] = selected_threshold
    best_checkpoint["threshold_scores"] = threshold_scores
    torch.save(best_checkpoint, output_dir / "best.pt")
    test_dataset = make_dataset(config, "test")
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        device,
        loss_config,
        selected_threshold,
        probability_normalization,
    )
    summary = {
        "experiment_type": (
            "causal hindcast (no post-issue observations)"
            if config["data"].get("forcing_mode", "causal") == "causal"
            else "perfect-forcing hindcast"
        ),
        "operational_forecast": False,
        "device": str(device),
        "initialized_from": initialized_from,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "samples": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "best_epoch": int(best_checkpoint["epoch"]),
        "checkpoint_validation_at_training_threshold": best_checkpoint["metrics"],
        "selected_threshold": selected_threshold,
        "threshold_validation_macro_iou": threshold_scores,
        "validation_loss": calibrated_val_loss,
        "validation": calibrated_val_metrics,
        "test_loss": test_loss,
        "test": test_metrics,
        "elapsed_seconds": time.time() - start_time,
        "config": config,
    }
    (output_dir / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
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
