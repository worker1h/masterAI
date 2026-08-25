"""Aggregate validation-selected SU-Net SAR preprocessing ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def render_comparison_plot(rows: list[dict[str, object]], output: Path) -> None:
    """Render an exact, publication-style validation/test IoU comparison."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(row["experiment"]) for row in rows]
    validation = np.asarray([float(row["val_macro_iou"]) for row in rows])
    test = np.asarray([float(row["test_macro_iou"]) for row in rows])
    x = np.arange(len(labels))
    width = 0.36

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    bars_val = axis.bar(
        x - width / 2,
        validation,
        width,
        label="Validation macro IoU",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
    )
    bars_test = axis.bar(
        x + width / 2,
        test,
        width,
        label="Test macro IoU (report only)",
        color="#F2CF5B",
        edgecolor="black",
        linewidth=0.8,
        hatch="//",
    )
    axis.set_title("SU-Net-inspired SAR preprocessing ablation")
    axis.set_ylabel("Macro IoU")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, max(float(validation.max()), float(test.max())) * 1.28)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncols=2, loc="upper left")
    axis.bar_label(bars_val, fmt="%.4f", padding=3, fontsize=8)
    axis.bar_label(bars_test, fmt="%.4f", padding=3, fontsize=8)
    axis.text(
        1.0,
        -0.20,
        "Epoch and threshold selected using validation data only; test never used for selection.",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


EXPERIMENTS = (
    ("standard", Path("outputs/gff_vit_ablation_standard/run.json")),
    ("dB only", Path("outputs/gff_vit_ablation_db/run.json")),
    ("CLAHE only", Path("outputs/gff_vit_ablation_clahe_only/run.json")),
    ("dB + CLAHE dual", Path("outputs/gff_vit_sunet/run.json")),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gff_sunet_ablation_summary.json"),
    )
    args = parser.parse_args()

    rows = []
    for name, path in EXPERIMENTS:
        if not path.exists():
            raise FileNotFoundError(f"Missing completed ablation result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        validation = result["validation"]
        test = result["test"]
        rows.append(
            {
                "experiment": name,
                "sar_preprocessing": result["config"]["data"].get(
                    "sar_preprocessing", "standard"
                ),
                "spatial_channels": 6 if name == "dB + CLAHE dual" else 4,
                "parameters": int(result["parameters"]),
                "best_epoch": int(result["best_epoch"]),
                "selected_threshold": float(result["selected_threshold"]),
                "val_macro_iou": float(validation["macro_iou"]),
                "test_macro_iou": float(test["macro_iou"]),
                "test_macro_dice": float(test["macro_dice"]),
                "test_h24_iou": float(test["h24_iou"]),
                "test_h48_iou": float(test["h48_iou"]),
                "test_h72_iou": float(test["h72_iou"]),
                "test_h24_boundary_f1": float(test["h24_boundary_f1"]),
                "test_h48_boundary_f1": float(test["h48_boundary_f1"]),
                "test_h72_boundary_f1": float(test["h72_boundary_f1"]),
            }
        )

    best_validation = max(rows, key=lambda row: row["val_macro_iou"])
    summary = {
        "selection_rule": (
            "Each experiment independently selects epoch and threshold on validation; "
            "test is report-only."
        ),
        "best_by_validation": best_validation["experiment"],
        "experiments": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    render_comparison_plot(
        rows, args.output.with_name("gff_sunet_ablation_comparison.png")
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
