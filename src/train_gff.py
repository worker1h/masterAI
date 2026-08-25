"""Train and evaluate GFFHorizonFormer on 24/48/72-hour hindcasts."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def heatmap_confidence(
    probability: torch.Tensor,
    valid: torch.Tensor | None = None,
    quantile: float = 0.99,
) -> torch.Tensor:
    """Measure absolute heatmap confidence with a valid-pixel upper quantile."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"confidence quantile must be in [0, 1], received {quantile}")
    mask = torch.ones_like(probability, dtype=torch.bool) if valid is None else valid.bool()
    values = []
    for sample, sample_mask in zip(probability, mask):
        selected = sample[sample_mask]
        if selected.numel() == 0:
            values.append(torch.zeros((), device=probability.device, dtype=torch.float32))
        else:
            upper_tail = max(
                int(math.ceil((1.0 - quantile) * selected.numel())), 1
            )
            values.append(
                torch.topk(selected.float(), upper_tail, sorted=False).values.amin()
            )
    return torch.stack(values).reshape(-1, 1, 1, 1)


def postprocess_probability_maps(
    probability: torch.Tensor,
    valid: torch.Tensor | None = None,
    threshold: float = 0.5,
    postprocessing: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Apply one consistent probability postprocessing rule before prediction."""

    settings = postprocessing or {}
    mode = str(settings.get("probability_normalization", "none"))
    batch_size = probability.shape[0]
    if mode == "adaptive_low_confidence_minmax":
        statistic = str(settings.get("confidence_statistic", "valid_quantile"))
        if statistic != "valid_quantile":
            raise ValueError(f"Unsupported confidence statistic: {statistic}")
        confidence = heatmap_confidence(
            probability,
            valid,
            float(settings.get("confidence_quantile", 0.99)),
        )
        confidence_gate = float(settings.get("confidence_gate", 0.1))
        normalized_maps = confidence < confidence_gate
        normalized_probability = normalize_probability_maps(
            probability, valid, "per_heatmap_minmax"
        )
        processed = torch.where(normalized_maps, normalized_probability, probability)
        raw_threshold = torch.full_like(
            confidence, float(settings.get("raw_threshold", threshold))
        )
        normalized_threshold = torch.full_like(
            confidence, float(settings.get("normalized_threshold", threshold))
        )
        thresholds = torch.where(
            normalized_maps, normalized_threshold, raw_threshold
        )
    else:
        processed = normalize_probability_maps(probability, valid, mode)
        normalized_maps = torch.full(
            (batch_size, 1, 1, 1),
            mode == "per_heatmap_minmax",
            dtype=torch.bool,
            device=probability.device,
        )
        confidence = torch.full(
            (batch_size, 1, 1, 1),
            float("nan"),
            dtype=torch.float32,
            device=probability.device,
        )
        thresholds = torch.full(
            (batch_size, 1, 1, 1),
            float(threshold),
            dtype=processed.dtype,
            device=processed.device,
        )
    return {
        "probability": processed,
        "prediction": processed >= thresholds,
        "normalized": normalized_maps,
        "confidence": confidence,
        "threshold": thresholds,
    }


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
def select_validation_adaptive_gate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    postprocessing: dict,
    candidates: list[float],
) -> tuple[float, dict[str, float], dict[str, dict[str, float]]]:
    """Select only the raw/normalized routing gate on validation data."""

    if str(postprocessing.get("probability_normalization")) != (
        "adaptive_low_confidence_minmax"
    ):
        raise ValueError("adaptive gate calibration requires adaptive postprocessing")
    if not candidates:
        raise ValueError("At least one confidence gate candidate is required")
    model.eval()
    totals = {
        gate: {horizon: np.zeros(4) for horizon in (1, 2, 3)}
        for gate in candidates
    }
    normalized_samples = {
        gate: {horizon: 0 for horizon in (1, 2, 3)} for gate in candidates
    }
    sample_counts = {horizon: 0 for horizon in (1, 2, 3)}
    quantile = float(postprocessing.get("confidence_quantile", 0.99))
    raw_threshold = float(postprocessing.get("raw_threshold", 0.3))
    normalized_threshold = float(postprocessing.get("normalized_threshold", 0.75))
    for batch in tqdm(loader, desc="calibrate adaptive gate", leave=False):
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
        probability = flood_probability(outputs)
        normalized_probability = normalize_probability_maps(
            probability, batch["valid"], "per_heatmap_minmax"
        )
        confidence = heatmap_confidence(probability, batch["valid"], quantile)
        raw_prediction = probability >= raw_threshold
        normalized_prediction = normalized_probability >= normalized_threshold
        truth = batch["target"] >= 0.5
        valid = batch["valid"] >= 0.5
        for horizon in (1, 2, 3):
            selected = batch["horizon"] == horizon
            sample_counts[horizon] += int(selected.sum())
            if not selected.any():
                continue
            target = truth[selected] & valid[selected]
            mask = valid[selected]
            for gate in candidates:
                use_normalized = confidence < gate
                prediction = torch.where(
                    use_normalized, normalized_prediction, raw_prediction
                )
                pred = prediction[selected] & mask
                totals[gate][horizon] += np.array(
                    [
                        (pred & target).sum().item(),
                        (pred & ~target & mask).sum().item(),
                        (~pred & target & mask).sum().item(),
                        (~pred & ~target & mask).sum().item(),
                    ]
                )
                normalized_samples[gate][horizon] += int(
                    use_normalized[selected].sum()
                )
    scores = {
        gate: float(
            np.mean(
                [confusion_metrics(counts)["iou"] for counts in by_horizon.values()]
            )
        )
        for gate, by_horizon in totals.items()
    }
    fractions = {
        str(gate): {
            f"h{horizon * 24}": normalized_samples[gate][horizon]
            / max(sample_counts[horizon], 1)
            for horizon in (1, 2, 3)
        }
        for gate in candidates
    }
    selected_gate = max(scores, key=scores.get)
    return (
        float(selected_gate),
        {str(key): value for key, value in scores.items()},
        fractions,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict,
    threshold: float = 0.5,
    probability_normalization: str = "none",
    postprocessing: dict | None = None,
) -> tuple[float, dict[str, float]]:
    model.eval()
    totals = {1: np.zeros(4), 2: np.zeros(4), 3: np.zeros(4)}
    boundary_totals = {1: np.zeros(3), 2: np.zeros(3), 3: np.zeros(3)}
    normalized_samples = {1: 0, 2: 0, 3: 0}
    sample_counts = {1: 0, 2: 0, 3: 0}
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
        effective_postprocessing = postprocessing or {
            "probability_normalization": probability_normalization
        }
        processed = postprocess_probability_maps(
            flood_probability(outputs),
            batch["valid"],
            threshold,
            effective_postprocessing,
        )
        processed_boundary = postprocess_probability_maps(
            torch.sigmoid(outputs["boundary"]),
            batch["valid"],
            threshold,
            effective_postprocessing,
        )
        prediction = processed["prediction"]
        boundary_prediction = processed_boundary["prediction"]
        target = batch["target"] >= 0.5
        boundary_target = batch["boundary"] >= 0.5
        valid = batch["valid"] >= 0.5
        for horizon in totals:
            selected = batch["horizon"] == horizon
            sample_counts[horizon] += int(selected.sum())
            normalized_samples[horizon] += int(
                processed["normalized"][selected].sum()
            )
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
    if str(
        (postprocessing or {}).get(
            "probability_normalization", probability_normalization
        )
    ) == "adaptive_low_confidence_minmax":
        for horizon in totals:
            metrics[f"h{horizon * 24}_normalized_fraction"] = float(
                normalized_samples[horizon] / max(sample_counts[horizon], 1)
            )
        metrics["normalized_fraction"] = float(
            sum(normalized_samples.values()) / max(sum(sample_counts.values()), 1)
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
    postprocessing = dict(config.get("postprocessing", {}))
    probability_normalization = str(
        postprocessing.get("probability_normalization", "none")
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
            postprocessing,
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
    adaptive = probability_normalization == "adaptive_low_confidence_minmax"
    if adaptive:
        selected_gate, gate_scores, normalized_fractions = (
            select_validation_adaptive_gate(
                model,
                val_loader,
                device,
                postprocessing,
                [
                    float(value)
                    for value in postprocessing["confidence_gate_candidates"]
                ],
            )
        )
        postprocessing["confidence_gate"] = selected_gate
        selected_threshold = float(postprocessing.get("raw_threshold", 0.3))
        threshold_scores = {}
    else:
        selected_threshold, threshold_scores = select_validation_threshold(
            model, val_loader, device, candidates, probability_normalization
        )
        gate_scores = {}
        normalized_fractions = {}
    calibrated_val_loss, calibrated_val_metrics = evaluate(
        model,
        val_loader,
        device,
        loss_config,
        selected_threshold,
        probability_normalization,
        postprocessing,
    )
    best_checkpoint["selected_threshold"] = selected_threshold
    if adaptive:
        best_checkpoint["selected_normalized_threshold"] = float(
            postprocessing["normalized_threshold"]
        )
        best_checkpoint["selected_confidence_gate"] = float(
            postprocessing["confidence_gate"]
        )
        best_checkpoint["confidence_gate_scores"] = gate_scores
    else:
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
        postprocessing,
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
        "selected_normalized_threshold": (
            float(postprocessing["normalized_threshold"]) if adaptive else None
        ),
        "selected_confidence_gate": (
            float(postprocessing["confidence_gate"]) if adaptive else None
        ),
        "probability_normalization": probability_normalization,
        "postprocessing": postprocessing,
        "threshold_validation_macro_iou": threshold_scores,
        "confidence_gate_validation_macro_iou": gate_scores,
        "confidence_gate_normalized_fractions": normalized_fractions,
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
