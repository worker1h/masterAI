"""Tkinter desktop UI for GFF flood-extent inference.

The selected Sentinel-1 file identifies a GFF scene.  The dataset loader then
retrieves the matching DEM, HAND and meteorological inputs required by the
trained model; an arbitrary RGB image is intentionally not accepted.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch

from src.gff_data import GFFFloodForecastDataset, WEATHER_CHANNELS
from src.gff_model import build_gff_model
from src.train_gff import flood_probability, postprocess_probability_maps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PredictionResult:
    preview: Image.Image
    mask: Image.Image
    probability: Image.Image
    metadata: dict


def detect_image_split(data_root: Path, image: Path, fold: int) -> tuple[str, Path]:
    """Return the configured train/val/test split and metadata for an S1 scene."""
    rois = data_root / "rois"
    selected = image.resolve()
    matching_meta: Path | None = None
    for meta_path in rois.glob("*-meta.json"):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if GFFFloodForecastDataset.component_paths(meta_path, metadata)["s1"].resolve() == selected:
                matching_meta = meta_path
                break
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue
    if matching_meta is None:
        raise ValueError("没有找到引用所选 Sentinel-1 图像的 GFF 元数据")

    source_fold: int | None = None
    for candidate in range(5):
        partition = data_root / "partitions" / f"floodmap_partition_{candidate}.txt"
        if not partition.is_file():
            continue
        names = {Path(line.strip()).name for line in partition.read_text(encoding="utf-8").splitlines() if line.strip()}
        if matching_meta.name in names:
            source_fold = candidate
            break
    if source_fold is None:
        raise ValueError(f"{matching_meta.name} 未出现在官方 partition 清单中")
    if source_fold == fold:
        split = "test"
    elif source_fold == (fold + 1) % 5:
        split = "val"
    else:
        split = "train"
    return split, matching_meta


def _dataset_kwargs(config: dict, split: str, root: Path) -> dict:
    data = config["data"]
    return {
        "root": root,
        "split": split,
        "fold": int(data.get("fold", 0)),
        "horizons": tuple(data.get("horizons", [1, 2, 3])),
        "weather_window": int(data.get("weather_window", 20)),
        "tile_size": int(data.get("tile_size", 224)),
        "context_size": int(data.get("context_size", 16)),
        "context_buffer_m": float(data.get("context_buffer_m", 50_000.0)),
        # The UI must be able to locate any selected scene, not just an
        # experiment's capped validation subset.
        "max_tiles": None,
        "max_sites": None,
        "seed": int(config.get("seed", 1337)),
        "augment": False,
        "forcing_mode": str(data.get("forcing_mode", "causal")),
        "sar_preprocessing": str(data.get("sar_preprocessing", "standard")),
        "sar_db_ranges": tuple(
            tuple(float(x) for x in pair)
            for pair in data.get("sar_db_ranges", [[-25.0, 0.0], [-32.0, -5.0]])
        ),
        "clahe_clip_limit": float(data.get("clahe_clip_limit", 2.0)),
        "clahe_grid_size": int(data.get("clahe_grid_size", 8)),
        "clahe_enhancement_size": int(data.get("clahe_enhancement_size", 256)),
    }


def _display_sar(spatial: torch.Tensor, preprocessing: str) -> np.ndarray:
    vv = spatial[0].detach().cpu().float().numpy()
    if preprocessing.startswith("sunet_"):
        unit = (np.clip(vv, -1.0, 1.0) + 1.0) * 0.5
    else:
        low, high = np.nanpercentile(vv, [2, 98])
        unit = (vv - low) / max(float(high - low), 1e-6)
    return np.uint8(np.clip(unit, 0.0, 1.0) * 255)


def _colour_probability(probability: np.ndarray) -> np.ndarray:
    """Small dependency-free blue/cyan/yellow probability colour map."""
    p = np.clip(probability, 0.0, 1.0)
    red = np.clip(2.2 * p - 0.8, 0.0, 1.0)
    green = np.clip(2.0 * p, 0.0, 1.0)
    blue = np.clip(1.4 - 1.4 * p, 0.0, 1.0)
    return np.uint8(np.stack([red, green, blue], axis=-1) * 255)


class FloodInferenceApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("洪析先知 · 洪水范围预测")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)

        self.checkpoint_path: Path | None = None
        self.image_path: Path | None = None
        self.checkpoint: dict | None = None
        self.config: dict | None = None
        self.model: torch.nn.Module | None = None
        self.dataset: GFFFloodForecastDataset | None = None
        self.matching_tiles: list[int] = []
        self.result: PredictionResult | None = None
        self.resource_signature: tuple[Path, Path, str] | None = None
        self.photos: list[ImageTk.PhotoImage] = []
        self.events: queue.Queue = queue.Queue()

        self.split = tk.StringVar(value="自动识别")
        self.horizon = tk.IntVar(value=72)
        self.tile_number = tk.IntVar(value=1)
        self.status = tk.StringVar(value="请选择模型权重和数据集中的 Sentinel-1 图像")
        self.device_text = tk.StringVar(
            value=f"计算设备：{'CUDA GPU' if torch.cuda.is_available() else 'CPU'}"
        )
        self._build()
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 22, "bold"))
        style.configure("Sub.TLabel", foreground="#5b6573")
        style.configure("Card.TLabelframe", padding=14)
        style.configure("Action.TButton", font=("Microsoft YaHei UI", 11, "bold"))

        shell = ttk.Frame(self.root, padding=22)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="洪析先知", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            shell,
            text="加载训练权重，预测选定区域未来新增洪水范围",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 16))

        controls = ttk.LabelFrame(shell, text="推理设置", style="Card.TLabelframe")
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)
        ttk.Button(controls, text="1  加载模型权重", command=self.choose_checkpoint).grid(row=0, column=0, sticky="w")
        self.weight_label = ttk.Label(controls, text="未选择", style="Sub.TLabel")
        self.weight_label.grid(row=0, column=1, sticky="ew", padx=12)
        ttk.Button(controls, text="2  浏览数据集图像", command=self.choose_image).grid(row=1, column=0, sticky="w", pady=10)
        self.image_label = ttk.Label(controls, text="未选择", style="Sub.TLabel")
        self.image_label.grid(row=1, column=1, sticky="ew", padx=12)

        options = ttk.Frame(controls)
        options.grid(row=2, column=0, columnspan=2, sticky="ew")
        ttk.Label(options, text="数据划分").pack(side="left")
        ttk.Label(options, textvariable=self.split, foreground="#1677a6").pack(side="left", padx=(6, 20))
        ttk.Label(options, text="预测时效").pack(side="left")
        ttk.Combobox(options, textvariable=self.horizon, values=(24, 48, 72), state="readonly", width=7).pack(side="left", padx=(6, 20))
        ttk.Label(options, text="场景瓦片").pack(side="left")
        self.tile_spin = ttk.Spinbox(options, from_=1, to=1, textvariable=self.tile_number, width=7, command=self.show_selected_tile)
        self.tile_spin.pack(side="left", padx=(6, 8))
        self.tile_count = ttk.Label(options, text="/ 0")
        self.tile_count.pack(side="left")
        ttk.Label(options, textvariable=self.device_text).pack(side="right")

        self.predict_button = ttk.Button(controls, text="3  开始预测", style="Action.TButton", command=self.start_prediction)
        self.predict_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0), ipady=7)

        results = ttk.Frame(shell)
        results.pack(fill="both", expand=True, pady=16)
        results.columnconfigure((0, 1, 2), weight=1, uniform="preview")
        results.rowconfigure(0, weight=1)
        self.image_panels = []
        for column, title in enumerate(("灾前 Sentinel-1", "洪水概率热力图", "新增洪水预测掩膜")):
            card = ttk.LabelFrame(results, text=title, padding=8)
            card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0))
            panel = ttk.Label(card, text="等待输入", anchor="center")
            panel.pack(fill="both", expand=True)
            self.image_panels.append(panel)

        footer = ttk.Frame(shell)
        footer.pack(fill="x")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=150)
        self.progress.pack(side="left")
        ttk.Label(footer, textvariable=self.status).pack(side="left", padx=10)
        self.save_button = ttk.Button(footer, text="保存预测结果", state="disabled", command=self.save_result)
        self.save_button.pack(side="right")

    def choose_checkpoint(self) -> None:
        value = filedialog.askopenfilename(
            title="选择 PyTorch 模型权重",
            initialdir=str(PROJECT_ROOT / "outputs"),
            filetypes=(("PyTorch 权重", "*.pt *.pth *.ckpt"), ("所有文件", "*.*")),
        )
        if value:
            self.checkpoint_path = Path(value)
            self.weight_label.configure(text=str(self.checkpoint_path))
            self.model = None
            self.dataset = None
            self.resource_signature = None
            self.status.set("权重已选择；首次预测时将完成加载")

    def choose_image(self) -> None:
        initial = PROJECT_ROOT / "data" / "gff" / "rois"
        value = filedialog.askopenfilename(
            title="选择 GFF Sentinel-1 图像",
            initialdir=str(initial if initial.exists() else PROJECT_ROOT / "data"),
            filetypes=(("GeoTIFF", "*.tif *.tiff"),),
        )
        if value:
            self.image_path = Path(value).resolve()
            self.image_label.configure(text=str(self.image_path))
            self.dataset = None
            self.matching_tiles = []
            self.resource_signature = None
            self.split.set("自动识别")
            self.tile_count.configure(text="/ 待索引")
            self.status.set("图像已选择；点击开始预测")

    def _set_busy(self, busy: bool) -> None:
        self.predict_button.configure(state="disabled" if busy else "normal")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def start_prediction(self) -> None:
        if not self.checkpoint_path or not self.image_path:
            messagebox.showwarning("输入不完整", "请先选择模型权重和 GFF Sentinel-1 图像。")
            return
        self._set_busy(True)
        self.status.set("正在加载数据与模型，请稍候…")
        request = {
            "checkpoint": self.checkpoint_path.resolve(),
            "image": self.image_path.resolve(),
            "horizon_hours": int(self.horizon.get()),
            "tile_position": max(int(self.tile_number.get()) - 1, 0),
        }
        threading.Thread(target=self._predict_worker, args=(request,), daemon=True).start()

    def _load_resources(self, request: dict) -> None:
        if (
            self.resource_signature is not None
            and self.resource_signature[:2] == (request["checkpoint"], request["image"])
            and self.model is not None
            and self.dataset is not None
        ):
            request["detected_split"] = self.resource_signature[2]
            return
        self.checkpoint = torch.load(request["checkpoint"], map_location="cpu", weights_only=False)
        config = self.checkpoint.get("config")
        if not isinstance(config, dict) or "data" not in config:
            raise ValueError("权重中没有完整 config；请使用本项目训练脚本生成的 checkpoint")
        self.config = config
        configured_root = Path(config["data"].get("root", "data/gff"))
        data_root = configured_root if configured_root.is_absolute() else PROJECT_ROOT / configured_root
        # A selected file from another GFF copy takes precedence when its
        # standard .../rois/<scene>.tif layout is recognizable.
        if request["image"].parent.name == "rois":
            data_root = request["image"].parent.parent
        split, _ = detect_image_split(data_root, request["image"], int(config["data"].get("fold", 0)))
        signature = (request["checkpoint"], request["image"], split)
        request["detected_split"] = split
        self.dataset = GFFFloodForecastDataset(**_dataset_kwargs(config, split, data_root))

        selected = request["image"]
        matches: list[int] = []
        for i, tile in enumerate(self.dataset.tiles):
            metadata = json.loads(tile.meta_path.read_text(encoding="utf-8"))
            s1 = self.dataset.component_paths(tile.meta_path, metadata)["s1"].resolve()
            if s1 == selected:
                matches.append(i)
        if not matches:
            raise ValueError(
                "所选图像不属于当前数据划分，或不是 GFF 的 Sentinel-1 输入。"
                "请切换 train/val/test 后重试。"
            )
        self.matching_tiles = matches

        # A full checkpoint contains every backbone parameter.  Disabling the
        # constructor's ImageNet initialization avoids needless network/cache
        # access before those parameters are immediately replaced.
        build_config = deepcopy(config)
        build_config.setdefault("model", {})["pretrained"] = False
        self.model = build_gff_model(build_config, self.dataset.spatial_channels, WEATHER_CHANNELS)
        self.model.load_state_dict(self.checkpoint["model"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.resource_signature = signature

    def _predict_worker(self, request: dict) -> None:
        try:
            self._load_resources(request)
            assert self.dataset and self.model and self.checkpoint and self.config
            tile_position = min(request["tile_position"], len(self.matching_tiles) - 1)
            tile_index = self.matching_tiles[tile_position]
            horizon_days = request["horizon_hours"] // 24
            if horizon_days not in self.dataset.horizons:
                raise ValueError(f"当前模型未训练 {horizon_days * 24} 小时时效")
            sample_index = tile_index * len(self.dataset.horizons) + self.dataset.horizons.index(horizon_days)
            sample = self.dataset[sample_index]
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
            ):
                outputs = self.model(
                    sample["spatial"][None].to(self.device),
                    sample["weather"][None].to(self.device),
                    sample["horizon"][None].to(self.device),
                    sample["forecast_mask"][None].to(self.device),
                )
            raw = flood_probability(outputs)
            threshold = float(self.checkpoint.get("selected_threshold", self.config.get("threshold", 0.5)))
            postprocessing = dict(self.config.get("postprocessing", {}))
            if "selected_confidence_gate" in self.checkpoint:
                postprocessing["confidence_gate"] = float(self.checkpoint["selected_confidence_gate"])
            processed = postprocess_probability_maps(raw, sample["valid"][None].to(self.device), threshold, postprocessing)
            probability = processed["probability"][0, 0].cpu().float().numpy()
            prediction = processed["prediction"][0, 0].cpu().numpy().astype(bool)
            valid = sample["valid"][0].numpy() >= 0.5
            prediction &= valid
            sar = _display_sar(sample["spatial"], str(self.config["data"].get("sar_preprocessing", "standard")))
            base = np.repeat(sar[..., None], 3, axis=-1)
            overlay = base.copy()
            overlay[prediction] = (30, 215, 225)
            preview = Image.fromarray(np.uint8(base * 0.45 + overlay * 0.55))
            mask = Image.fromarray(np.uint8(prediction) * 255)
            heatmap = Image.fromarray(_colour_probability(probability))
            metadata = {
                "source_image": str(request["image"]),
                "checkpoint": str(request["checkpoint"]),
                "site": sample["site"],
                "dataset_split": request["detected_split"],
                "tile": tile_position + 1,
                "matching_tiles": len(self.matching_tiles),
                "horizon_hours": horizon_days * 24,
                "threshold": float(processed["threshold"][0].item()),
                "predicted_flood_pixels": int(prediction.sum()),
                "predicted_flood_ratio": float(prediction.sum() / max(valid.sum(), 1)),
            }
            self.events.put(("success", PredictionResult(preview, mask, heatmap, metadata)))
        except Exception as exc:
            self.events.put(("error", f"{exc}\n\n{traceback.format_exc()}"))

    def _poll_events(self) -> None:
        try:
            kind, payload = self.events.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_events)
            return
        self._set_busy(False)
        if kind == "error":
            self.status.set("预测失败")
            messagebox.showerror("预测失败", payload)
        else:
            self.result = payload
            self.split.set(payload.metadata["dataset_split"])
            self.tile_spin.configure(to=max(len(self.matching_tiles), 1))
            self.tile_count.configure(text=f"/ {len(self.matching_tiles)}")
            self._show_images(payload)
            self.save_button.configure(state="normal")
            ratio = payload.metadata["predicted_flood_ratio"] * 100
            self.status.set(f"预测完成 · {payload.metadata['horizon_hours']} h · 新增洪水像素占比 {ratio:.2f}%")
        self.root.after(100, self._poll_events)

    def _show_images(self, result: PredictionResult) -> None:
        images = (result.preview, result.probability, result.mask)
        self.photos.clear()
        for panel, source in zip(self.image_panels, images):
            shown = source.resize((320, 320), Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(shown)
            self.photos.append(photo)
            panel.configure(image=photo, text="")

    def show_selected_tile(self) -> None:
        if self.result and self.checkpoint_path and self.image_path:
            self.start_prediction()

    def save_result(self) -> None:
        if not self.result:
            return
        value = filedialog.asksaveasfilename(
            title="保存预测结果",
            initialfile=f"{self.result.metadata['site']}_{self.result.metadata['horizon_hours']}h_prediction.png",
            defaultextension=".png",
            filetypes=(("PNG 图像", "*.png"),),
        )
        if not value:
            return
        output = Path(value)
        canvas = Image.new("RGB", (960, 360), "white")
        draw = ImageDraw.Draw(canvas)
        for x, image, title in zip((0, 320, 640), (self.result.preview, self.result.probability, self.result.mask), ("Sentinel-1 + forecast", "Flood probability", "Flood mask")):
            canvas.paste(image.convert("RGB").resize((320, 320), Image.Resampling.NEAREST), (x, 0))
            draw.text((x + 8, 330), title, fill="black")
        canvas.save(output)
        self.result.mask.save(output.with_name(output.stem + "_mask.png"))
        output.with_suffix(".json").write_text(json.dumps(self.result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.set(f"结果已保存：{output}")


def main() -> None:
    root = tk.Tk()
    FloodInferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
