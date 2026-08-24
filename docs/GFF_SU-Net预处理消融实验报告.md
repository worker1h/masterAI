# GFF SU-Net 风格 SAR 预处理消融实验报告

## 1. 目的

本实验不再同时更换预处理和模型，而是在完全相同的 `GFFViTHorizonFormer` 上只改变 Sentinel-1 VV/VH 的输入表达，回答以下问题：

1. 线性后向散射转固定范围 dB 是否有效；
2. CLAHE 单独使用是否有效；
3. SU-Net 的“原图 + CLAHE 图”双视图是否优于单一输入；
4. 改善是否能从 validation 泛化到独立流域 test。

SU-Net 原论文用于非合作航天器 ISAR 位姿估计，本实验只迁移其保留原图并增加 CLAHE 图的输入思想。GFF 是 Sentinel-1 SAR，因此在 CLAHE 之前增加了线性后向散射到 dB 的物理转换。

## 2. 消融组

| 组别 | SAR 输入 | SAR 通道 | 空间总通道 | 作用 |
|---|---|---:|---:|---|
| Standard | 官方线性 VV/VH z-score | 2 | 4 | 无 SU-Net 处理的控制组 |
| dB only | 固定范围 VV/VH dB | 2 | 4 | 隔离 dB 转换贡献 |
| CLAHE only | dB 后仅保留 VV/VH CLAHE | 2 | 4 | 隔离增强图贡献 |
| dB + CLAHE dual | 原始 dB 与 CLAHE VV/VH | 4 | 6 | 完整 SU-Net 风格双视图 |

四组均拼接 DEM/HAND。dB 组统一使用 VV `[-25, 0] dB`、VH `[-32, -5] dB`；CLAHE 统一在 256×256 上使用 clip limit 2.0、8×8 网格，再恢复到 224×224。

## 3. 控制变量与选择协议

- 数据固定 seed 1337，train/val/test 分别使用相同 3,000/500/500 瓦片；
- fold 2–4 训练、fold 1 验证、fold 0 测试；
- 同一 ImageNet 预训练 ViT-B/16、冻结前 6 个 block、相同 U 形解码器和时间 Transformer；
- 同一严格因果 24/48/72h 输入、均衡采样、Focal-Dice、边界及存在性损失；
- batch 4、梯度累积 4、4 轮、AdamW 和同一学习率；
- 每组分别在 validation 上对四轮 checkpoint 和 0.1–0.7 阈值做选择；
- test 只在选择完成后评估，不用于挑组、轮次或阈值。

四通道组和六通道组的首层卷积形状不同，但都由同一预训练 RGB patch projection 的通道均值扩展，并按 `3 / 输入通道数` 缩放。这是双视图增加输入通道不可避免的结构差异，报告会同时列出参数量。

## 4. 正式结果

### 4.1 验证集选择与独立测试

| 组别 | 空间通道 | 参数量 | 最佳 epoch | 阈值 | Val Macro IoU | Test Macro IoU | Test Macro Dice | Test Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard | 4 | 89,171,147 | 3 | 0.5 | 0.038424 | 0.030759 | 0.059681 | 0.000102 |
| **dB only** | 4 | 89,171,147 | 2 | 0.3 | **0.044264** | **0.050801** | **0.096690** | **0.022058** |
| CLAHE only | 4 | 89,171,147 | 3 | 0.5 | 0.042379 | 0.030713 | 0.059595 | 0.002184 |
| dB + CLAHE dual | 6 | 89,564,939 | 3 | 0.5 | 0.043690 | 0.035029 | 0.067686 | 0.006445 |

表中的 Boundary F1 是 24/48/72h 三个时效的宏平均。按预先确定的 validation 选择规则，最终胜出组是 **dB only**，而不是 test 上偶然出现较高数值的检查点。

### 4.2 不同时效测试 IoU

| 组别 | 24h IoU | 48h IoU | 72h IoU |
|---|---:|---:|---:|
| Standard | 0.030981 | 0.030997 | 0.030299 |
| **dB only** | **0.050849** | **0.050469** | **0.051085** |
| CLAHE only | 0.030746 | 0.030209 | 0.031184 |
| dB + CLAHE dual | 0.034117 | 0.035447 | 0.035523 |

![SU-Net 风格 SAR 预处理消融对比](../outputs/gff_sunet_ablation_comparison.png)

### 4.3 结果解释

1. **固定范围 dB 转换是本轮最有效的预处理。** 相对 Standard，dB only 的 validation Macro IoU 绝对提高 0.005840（相对 +15.2%），独立 test 绝对提高 0.020042（相对 +65.2%）。三个时效均得到一致提升，测试边界 F1 也从 0.000102 提高到 0.022058。
2. **单独 CLAHE 没有形成稳定跨流域收益。** 它的 validation Macro IoU 相对 Standard 提高 10.3%，但 test Macro IoU 反而低 0.15%。这说明增强图能改善验证域拟合，却没有在当前小样本训练预算下可靠泛化。
3. **双视图优于 Standard，但不如只用 dB。** dB + CLAHE dual 相对 Standard 的 validation/test Macro IoU 分别提高 13.7%/13.9%；但相对 dB only 分别低 1.3%/31.0%，且多 393,792 个参数。因此当前实验不支持把 CLAHE 双视图保留在主模型中。
4. **测试集不能反向参与选模。** CLAHE only 第 4 轮、阈值 0.4 曾得到 test Macro IoU 0.072805，但其 validation Macro IoU 只有 0.039034；联合验证选择最终选中第 3 轮、阈值 0.5（validation 0.042379，test 0.030713）。正式表只报告后者，避免 test 泄漏和结果挑选。

最终机器可读结果保存为：

- `outputs/gff_sunet_ablation_summary.json`
- `outputs/gff_sunet_ablation_summary.csv`
- `outputs/gff_sunet_ablation_comparison.png/pdf`

## 5. 复现

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gff_sunet_ablation.ps1
```

脚本依次运行专项测试、Standard、dB only、CLAHE only 三组训练与逐轮校准，并复用已经完成的 dB + CLAHE dual 结果生成汇总。全流程只使用现有 conda `daily` 环境。

## 6. 胜出模型全训练折微调

消融完成后，以 dB only 的最佳第 2 轮权重初始化模型，只加载模型参数并重新初始化优化器和余弦学习率调度器。在 fold 2–4 的全部 98,853 个训练瓦片（296,559 个“瓦片×时效”样本）上微调 2 个完整 epoch；逐轮验证和最终测试分别使用固定 2,000 个瓦片（各 6,000 个时效样本）。微调学习率降为任务头 `3e-5`、ViT backbone `3e-6`。

| 指标 | Epoch 1 | Epoch 2 / 最终选择 |
|---|---:|---:|
| 固定阈值 0.3 Val Macro IoU | 0.044425 | **0.058788** |
| 固定阈值 0.3 Val Macro Dice | 0.085070 | **0.111048** |
| 联合选择阈值 | 0.4 | **0.3** |
| 联合选择 Val Macro IoU | 0.051018 | **0.058788** |

验证集联合选择第 2 轮、阈值 0.3。独立测试结果如下：

| Test 指标 | 24h | 48h | 72h | Macro |
|---|---:|---:|---:|---:|
| IoU | 0.063057 | 0.063254 | 0.063348 | **0.063220** |
| Dice | — | — | — | **0.118921** |
| Boundary F1 | 0.064960 | 0.064462 | 0.062983 | **0.064135** |

本次完整训练折微调耗时约 7 小时 25 分钟，训练和验证过程无 NaN、显存溢出或数据读取错误。结果文件位于 `outputs/gff_vit_db_full_finetune/`。自动预测样例的 72h IoU 为 0，它是一个真实漏检失败案例，应与总体指标一起保留，而不能只展示事后挑选的最佳案例。

需要注意：完整训练实验使用 2,000/2,000 个固定验证/测试瓦片，而消融使用 500/500 个瓦片，因此两者数值不能视为只改变训练数据量的严格配对比较。

## 7. 逐热度图概率归一化与阈值复算

按“每一张热度图先归一化、再计算阈值”的要求，新增 `per_heatmap_minmax` 后处理。对每个样本的有效像素独立计算

\[
\hat p_{ij}=\frac{p_{ij}-p_{\min}}{p_{\max}-p_{\min}},\qquad
\hat y_{ij}=\mathbb{1}(\hat p_{ij}\ge \tau).
\]

无效像素始终置零；若一张有效热度图没有动态范围（`p_max - p_min <= 1e-6`），归一化结果也置零，避免除零和把常数噪声误判为洪水。分割概率图和边界概率图均采用相同规则。模型权重没有重新训练，仅复用全训练折的两个 checkpoint，在 validation 上重新联合选择 epoch 和归一化阈值，候选范围扩展为 0.1–0.95。

| 后处理 | 最佳 epoch | Val 选择阈值 | Val Macro IoU | Test Macro IoU | Test Macro Dice | Test Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| 原始概率（正式主结果） | 2 | 0.30 | **0.058788** | 0.063220 | 0.118921 | **0.064135** |
| 逐图 min–max（可选诊断） | 2 | 0.75 | 0.049237 | **0.091828** | **0.168210** | 0.019376 |

逐图归一化的测试 24/48/72h IoU 分别为 0.092195/0.091762/0.091528。它修复了默认 72h 样例的整体概率偏低：同一 site、同一 checkpoint 的原始概率范围仅为 0.001085–0.135864，原始阈值下 IoU 为 0；归一化后使用 validation 选择的阈值 0.75，IoU 提高到 0.810202（TP 6,147、FP 689、FN 751）。对比图保存在 `outputs/gff_vit_db_full_finetune_per_heatmap/prediction_72h.png`。

这个结果不能直接替代正式主结果：归一化使 validation Macro IoU 下降 16.25%，但 test Macro IoU 上升 45.25%，说明两个跨流域子集的概率尺度分布不同。测试集已经被查看，不能据此反向选择后处理。此外，逐图 min–max 会消除图块级绝对置信度和存在性软门控的幅值作用，只要热度图存在微小动态范围就会把最大值拉到 1，容易放大无洪水图块中的噪声；测试 Boundary F1 也明显下降。因此它作为可配置后处理和尺度漂移诊断保留，正式模型仍采用 validation 表现更高的原始概率方案。

复算配置与机器可读结果分别为：

- `configs/gff_vit_db_full_finetune_per_heatmap.yaml`
- `outputs/gff_vit_db_full_finetune_per_heatmap/run.json`
- `outputs/gff_vit_db_full_finetune_per_heatmap/best.pt`

## 8. 结论边界

这是一项固定小子集单随机种子消融，能够比较当前训练预算内的相对趋势，但不能替代完整 98,853 个训练瓦片和多随机种子统计。若 validation 与 test 排序不一致，应优先报告跨流域标定问题，而不能用 test 反向选择看起来最好的组。

当前正式模型采用 **VV/VH 固定范围 dB + DEM/HAND 的四通道输入**，暂不加入 CLAHE，并已完成全部训练折微调。下一阶段应在预算允许时补 3 个随机种子，并扩大验证/测试覆盖；只有结论稳定后，才值得继续搜索 CLAHE clip limit、网格尺度或融合位置。
