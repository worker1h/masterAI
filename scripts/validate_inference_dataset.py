"""Validate that a GFF directory contains the files required for inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rasterio

from src.gff_data import GFFFloodForecastDataset


NORMALISATION_FILES = (
    "s1_norm_0.csv", "dem_norm_0.csv", "hand_norm_0.csv", "glofas_norm_0.csv",
    "era5_norm.csv", "era5_land_norm.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/gff"))
    parser.add_argument("--image", type=Path, help="Optionally validate one selected S1 GeoTIFF")
    return parser.parse_args()


def validate(root: Path, image: Path | None = None) -> list[str]:
    errors: list[str] = []
    for name in NORMALISATION_FILES:
        if not (root / "normalisation" / name).is_file():
            errors.append(f"缺少标准化文件: normalisation/{name}")
    for fold in range(5):
        name = f"floodmap_partition_{fold}.txt"
        if not (root / "partitions" / name).is_file():
            errors.append(f"缺少划分文件: partitions/{name}")
    rois = root / "rois"
    if not rois.is_dir():
        return [*errors, "缺少目录: rois/"]

    if image is None:
        meta_paths = sorted(rois.glob("*-meta.json"))
        if not meta_paths:
            return [*errors, "rois/ 中没有 *-meta.json"]
    else:
        image = image.resolve()
        if not image.is_file():
            return [*errors, f"图像不存在: {image}"]
        if image.parent.resolve() != rois.resolve():
            errors.append("图像必须直接位于指定 root 的 rois/ 目录中")
        try:
            with rasterio.open(image) as source:
                if source.count < 2:
                    errors.append("Sentinel-1 GeoTIFF 必须至少包含 VV、VH 两个波段")
        except Exception as exc:
            errors.append(f"无法读取 GeoTIFF: {exc}")
        meta_paths = sorted(rois.glob("*-meta.json"))

    matched_image = image is None
    complete_sites = 0
    for meta_path in meta_paths:
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            required_keys = {"key", "pre1_date", "post_date", "visit_tiles", "floodmap"}
            absent_keys = sorted(required_keys - metadata.keys())
            if absent_keys:
                errors.append(f"{meta_path.name} 缺少字段: {', '.join(absent_keys)}")
                continue
            paths = GFFFloodForecastDataset.component_paths(meta_path, metadata)
            if image is not None and paths["s1"].resolve() != image:
                continue
            matched_image = True
            missing = [path.name for path in paths.values() if not path.is_file()]
            if missing:
                errors.append(f"{meta_path.name} 缺少配套文件: {', '.join(missing)}")
            else:
                complete_sites += 1
        except Exception as exc:
            errors.append(f"无法解析 {meta_path.name}: {exc}")
    if not matched_image:
        errors.append("没有元数据记录引用所选 Sentinel-1 图像")
    if complete_sites == 0:
        errors.append("没有找到文件齐全、可用于推理的站点")
    return errors


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    errors = validate(root, args.image)
    if errors:
        print("数据集校验失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"数据集校验通过: {root}")
    if args.image:
        print(f"可推理图像: {args.image.resolve()}")


if __name__ == "__main__":
    main()
