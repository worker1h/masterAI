import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import WeightedRandomSampler


def flood_fraction_bins(fractions, thresholds=(0.0, 0.01, 0.10)):
    """Map fractions to zero, low, medium and high-flood tile bins."""
    fractions = np.asarray(fractions, dtype=np.float64)
    return np.where(
        fractions <= thresholds[0],
        0,
        np.where(fractions < thresholds[1], 1, np.where(fractions < thresholds[2], 2, 3)),
    )


def balanced_bin_weights(bin_ids, target_probabilities=None):
    bin_ids = np.asarray(bin_ids, dtype=np.int64)
    n_bins = int(bin_ids.max()) + 1
    counts = np.bincount(bin_ids, minlength=n_bins)
    if target_probabilities is None:
        probabilities = np.full(n_bins, 1 / n_bins, dtype=np.float64)
    else:
        probabilities = np.asarray(target_probabilities, dtype=np.float64)
        if len(probabilities) != n_bins or np.any(probabilities < 0):
            raise ValueError(f"Expected {n_bins} non-negative target probabilities")
        probabilities = probabilities / probabilities.sum()
    if np.any((counts == 0) & (probabilities > 0)):
        raise ValueError("A requested sampling bin contains no tiles")
    weights = np.asarray([probabilities[b] / counts[b] for b in bin_ids], dtype=np.float64)
    return weights, counts.tolist(), probabilities.tolist()


def _read_flood_fraction(item):
    name, path = item
    with rasterio.open(path) as src:
        mask = src.read(1)
    return name, float(np.count_nonzero(mask == 2) / mask.size)


def dataset_flood_fractions(dataset, cache_path, workers=8):
    """Read flood-pixel fractions once and cache them by sample name."""
    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = [(name, dataset.mask_paths[name]) for name in dataset.names if name not in cache]
    if missing:
        print(f"Caching flood fractions for {len(missing)} tiles with {workers} readers...", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            for name, fraction in executor.map(_read_flood_fraction, missing, chunksize=64):
                cache[name] = fraction
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"Flood-fraction cache written to {cache_path}", flush=True)
    return np.asarray([cache[name] for name in dataset.names], dtype=np.float64)


def make_flood_fraction_sampler(dataset, config, seed, cache_path):
    fractions = dataset_flood_fractions(
        dataset, cache_path, workers=config.get("cache_workers", 8)
    )
    thresholds = tuple(config.get("thresholds", [0.0, 0.01, 0.10]))
    bins = flood_fraction_bins(fractions, thresholds)
    weights, counts, probabilities = balanced_bin_weights(
        bins, config.get("target_probabilities")
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(config.get("samples_per_epoch", len(dataset))),
        replacement=True,
        generator=generator,
    )
    metadata = {
        "strategy": "flood_fraction_bins",
        "thresholds": list(thresholds),
        "bin_counts": counts,
        "target_probabilities": probabilities,
        "mean_flood_fraction": float(fractions.mean()),
    }
    return sampler, metadata
