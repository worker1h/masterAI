"""Audit GFF split coverage, class imbalance and component availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gff_data import GFFFloodForecastDataset, read_tile_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/gff"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    rois = root / "rois"
    report = {"root": str(root.resolve()), "folds": {}}
    for fold in range(5):
        names = [
            line.strip()
            for line in (root / "partitions" / f"floodmap_partition_{fold}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        totals = {
            "listed_sites": len(names),
            "era5_eligible_sites": 0,
            "tiles": 0,
            "positive_tiles": 0,
            "valid_pixels": 0,
            "flooded_pixels": 0,
            "complete_sites": 0,
        }
        for name in names:
            meta_path = rois / Path(name).name
            if not meta_path.exists():
                continue
            totals["era5_eligible_sites"] += 1
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            paths = GFFFloodForecastDataset.component_paths(meta_path, metadata)
            totals["complete_sites"] += int(all(path.exists() for path in paths.values()))
            for _, background, permanent, flooded in read_tile_index(paths["geometry"]):
                totals["tiles"] += 1
                totals["positive_tiles"] += int(flooded > 0)
                totals["valid_pixels"] += background + permanent + flooded
                totals["flooded_pixels"] += flooded
        totals["positive_tile_ratio"] = totals["positive_tiles"] / max(totals["tiles"], 1)
        totals["flood_pixel_ratio"] = totals["flooded_pixels"] / max(
            totals["valid_pixels"], 1
        )
        report["folds"][str(fold)] = totals
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
