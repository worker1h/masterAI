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

from src.data import FutureEventForecastDataset
from src.model import build_model


def predict(model, inputs, device):
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        return torch.sigmoid(model(inputs[None].to(device)))[0, 0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = FutureEventForecastDataset(config["manifest"], args.partition)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], 4, config["base_channels"]).to(device)
    state = torch.load(
        Path(config["output_dir"]) / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(state["model"])
    model.eval()

    if args.sample:
        index = next(
            index
            for index, sample in enumerate(dataset.samples)
            if sample["future_name"] == args.sample
        )
    else:
        ranked = []
        for index in range(len(dataset)):
            inputs, target, _ = dataset[index]
            probability = predict(model, inputs, device)
            prediction = probability >= args.threshold
            truth = target[0].numpy() >= 0.5
            intersection = np.logical_and(prediction, truth).sum()
            union = np.logical_or(prediction, truth).sum()
            if truth.any():
                ranked.append((intersection / max(union, 1), index))
        ranked.sort()
        index = ranked[len(ranked) // 2][1]

    inputs, target, name = dataset[index]
    metadata = dataset.samples[index]
    probability = predict(model, inputs, device)
    truth = target[0].numpy() >= 0.5
    prediction = probability >= args.threshold
    true_positive = np.logical_and(truth, prediction)
    false_positive = np.logical_and(~truth, prediction)
    false_negative = np.logical_and(truth, ~prediction)
    intersection = true_positive.sum()
    union = np.logical_or(truth, prediction).sum()
    iou = intersection / max(union, 1)
    precision = intersection / max(prediction.sum(), 1)
    recall = intersection / max(truth.sum(), 1)

    pre_vv = inputs[0].numpy()
    event_vv = inputs[2].numpy()
    difference = event_vv - pre_vv
    joint = np.concatenate([pre_vv.ravel(), event_vv.ravel()])
    vv_min, vv_max = np.percentile(joint, [2, 98])
    difference_limit = max(float(np.percentile(np.abs(difference), 98)), 1e-6)
    comparison = np.ones((*truth.shape, 3), dtype=np.float32)
    comparison[true_positive] = np.asarray([55, 117, 186]) / 255
    comparison[false_positive] = np.asarray([182, 67, 66]) / 255
    comparison[false_negative] = np.asarray([224, 165, 56]) / 255
    mask_map = ListedColormap(["#FFFFFF", "#0F4D92"])

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    panels = [
        (pre_vv, "(a) 上一次洪灾前 VV", "gray", vv_min, vv_max),
        (event_vv, "(b) 上一次洪灾中 VV", "gray", vv_min, vv_max),
        (difference, "(c) 上一次洪灾时相变化", "coolwarm", -difference_limit, difference_limit),
        (truth, "(d) 下一次洪灾实际范围", mask_map, 0, 1),
        (probability, "(e) 下一次洪灾预测概率", "viridis", 0, 1),
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
    )
    history_date = metadata["history_event_time"][:10]
    future_date = metadata["future_event_time"][:10]
    figure.suptitle(
        f"{metadata['history_event']} ({history_date}) → {metadata['future_event']} ({future_date})\n"
        f"{name}   IoU={iou:.3f}   Precision={precision:.3f}   Recall={recall:.3f}",
        fontsize=14,
        y=0.98,
    )
    figure.subplots_adjust(left=0.03, right=0.97, top=0.89, bottom=0.09, wspace=0.08, hspace=0.18)
    output_dir = Path(
        args.output_dir or Path(config["output_dir"]) / "future_predictions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}_next_event_forecast.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
