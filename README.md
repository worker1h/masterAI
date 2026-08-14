# 洪析先知：ImpactMesh-Flood 实验基线

本仓库实现项目文档“阶段一”的最小可复现实验闭环：ImpactMesh-Flood 数据下载与核验、E0-E3 输入消融、U-Net 二分类分割、IoU/Dice/Precision/Recall/F1 评估、checkpoint 与可视化归档。

## 快速开始

```powershell
conda run -n daily python -m pip install -r requirements.txt
conda run -n daily python scripts\download_data.py --split val --modalities MASK DEM S1RTC
conda run -n daily python scripts\inspect_data.py --data-root data\raw --split val --limit 20
conda run -n daily python -m src.train --config configs\e0_smoke.yaml
```

正式实验将配置改为 `configs/e0.yaml`、`e1.yaml`、`e2.yaml`、`e3.yaml`。数据归档来自官方 Hugging Face 仓库，解包后目录应为 `data/raw/<split>/<modality>/*.zarr.zip`。

## 实验定义

- E0：event S1RTC (VV/VH)
- E1：event S1RTC + DEM
- E2：pre-event + event S1RTC
- E3：pre-event + event S1RTC + DEM

默认按官方 split 训练与验证；同一配置固定随机种子。`stage1_e*.yaml` 使用本地 validation 数据进行事件级留出可行性实验，`e*.yaml` 用于正式 train/val 实验。smoke 配置只验证链路，不作为竞赛指标。

## 当前阶段一结果

正式 train→val 的 E0/E1/E2/E3 IoU 为 0.4774/0.5066/0.5691/0.5614；test_holdout IoU 为 0.3616/0.3803/0.4941/0.4697。最优为 E2（pre-event + event SAR）。详见 `docs/第一阶段实验报告.md` 与 `outputs/formal_summary/formal_ablation.csv`。

完整复现实验可运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_formal_experiments.ps1
```

## 初赛提交文件

生成与校验材料：

```powershell
conda run -n daily python scripts\build_submission.py
conda run -n daily python scripts\build_video.py
conda run -n daily python scripts\validate_submission.py
```

最终四项文件位于 `output/submission/`：参赛作品简介 PDF、项目文档 PDF、项目视频 MP4 和其他材料 ZIP。`洪析先知团队` 是当前占位团队名，正式提交前须改为报名系统中的准确团队名称，并重新运行以上脚本。

## 类别不平衡与边界实验

- `formal_e2_imbalance_boundary.yaml`：分洪水占比重采样 + Focal/Tversky + 边界带加权。该组合用于验证“重复正类补偿”的风险，不能默认视为改进。
- `formal_e2_boundary_precision.yaml`：保留原始图块分布，将正类权重从 4.0 降为 2.0，并仅在五像素边界带上提高 BCE 权重，用于降低误报并改善边缘定位。
