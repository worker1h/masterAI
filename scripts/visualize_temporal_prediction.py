import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CHANNELS, ImpactMeshDataset
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if config["input_mode"] not in ("e2", "e3"):
        raise ValueError("Temporal visualization requires an E2/E3 pre-event and event input")
    dataset = ImpactMeshDataset(
        config["data_root"], args.split, config["input_mode"], sample_names=[args.sample]
    )
    inputs, target, name = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config.get("model", "unet"), CHANNELS[config["input_mode"]], config["base_channels"]
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(inputs[None].to(device)))[0, 0].cpu().numpy()

    pre_vv = inputs[0].numpy()
    event_vv = inputs[2].numpy()
    difference = event_vv - pre_vv
    truth = target[0].numpy() >= 0.5
    prediction = probability >= 0.5
    true_positive = np.logical_and(truth, prediction)
    false_positive = np.logical_and(~truth, prediction)
    false_negative = np.logical_and(truth, ~prediction)
    union = np.logical_or(truth, prediction).sum()
    intersection = true_positive.sum()
    iou = intersection / max(union, 1)
    precision = intersection / max(prediction.sum(), 1)
    recall = intersection / max(truth.sum(), 1)

    joint = np.concatenate([pre_vv.ravel(), event_vv.ravel()])
    vv_min, vv_max = np.percentile(joint, [2, 98])
    difference_limit = max(float(np.percentile(np.abs(difference), 98)), 1e-6)
    actual_map = ListedColormap(["#FFFFFF", "#0F4D92"])
    predicted_map = ListedColormap(["#FFFFFF", "#B64342"])
    comparison = np.ones((*truth.shape, 3), dtype=np.float32)
    comparison[true_positive] = np.asarray([55, 117, 186]) / 255
    comparison[false_positive] = np.asarray([182, 67, 66]) / 255
    comparison[false_negative] = np.asarray([224, 165, 56]) / 255

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    panels = [
        (pre_vv, "(a) 灾前 VV", "gray", vv_min, vv_max),
        (event_vv, "(b) 灾中 VV", "gray", vv_min, vv_max),
        (difference, "(c) 灾中 − 灾前", "coolwarm", -difference_limit, difference_limit),
        (truth, "(d) 实际新增洪水", actual_map, 0, 1),
        (prediction, "(e) 预测新增洪水", predicted_map, 0, 1),
        (comparison, "(f) 预测与实际对比", None, None, None),
    ]
    for axis, (image, title, color_map, minimum, maximum) in zip(axes.flat, panels):
        axis.imshow(image, cmap=color_map, vmin=minimum, vmax=maximum)
        axis.set_title(title, fontsize=12, pad=8)
        axis.axis("off")
    axes[1, 2].legend(
        handles=[
            Patch(facecolor="#3775BA", label="正确识别"),
            Patch(facecolor="#B64342", label="误报"),
            Patch(facecolor="#E0A538", label="漏检"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    figure.suptitle(
        f"{name}\nIoU={iou:.3f}   Precision={precision:.3f}   Recall={recall:.3f}",
        fontsize=14,
        y=0.98,
    )
    figure.subplots_adjust(left=0.03, right=0.97, top=0.89, bottom=0.09, wspace=0.08, hspace=0.18)
    output_dir = Path(args.output_dir or Path(config["output_dir"]) / "predictions_temporal")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_temporal_comparison.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
