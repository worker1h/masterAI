# GFF ViT + SU-Net 风格 SAR 增强实验报告

## 1. 实验目标

在不改变 GFF 严格因果 24/48/72 小时洪水范围预测定义的前提下，针对原始 Sentinel-1 输入对比度低、细节不直观的问题，参考 SU-Net 的 ISAR 预处理思路保留原图并加入 CLAHE 增强图，同时把空间编码器由 MiT-B0 调整为 ImageNet 预训练 ViT-B/16。实验仍只输入目标时刻以前可用的信息，不读取目标时刻或灾后的 Sentinel-1。

SU-Net 论文：https://www.nature.com/articles/s41598-023-38974-1
SU-Net 官方代码：https://github.com/Tombs98/SU-Net

SU-Net 本身解决非合作航天器 ISAR 位姿估计，不是洪水预测模型。本实验迁移的是“原始图 + CLAHE 增强图”的输入处理方式，不照搬其标签、损失或完整网络。

## 2. SAR 预处理

GFF Sentinel-1 数据是线性后向散射值，不能直接当作普通 8-bit ISAR 显示图做直方图均衡。新流水线按下列顺序处理：

1. 对 VV/VH 分别执行 `10 × log10(max(x, 1e-6))`，转换为 dB；
2. VV 固定裁剪到 `[-25, 0] dB`，VH 固定裁剪到 `[-32, -5] dB`，映射至 `[-1, 1]`；
3. 按 SU-Net 的方式在 256×256 尺寸执行 CLAHE（clip limit 2.0、8×8 网格），再还原到 224×224；
4. 同时保留原始 dB 与增强版本，组成 `VV + VH + VV-CLAHE + VH-CLAHE`；
5. 拼接 DEM 与 HAND，空间输入共 6 通道。

对真实 GFF 样本抽查，原始 dB 通道标准差为 0.1160/0.1142，CLAHE 通道为 0.2531/0.2565；增强前后平均绝对差为 0.1231/0.1319。由此确认增强并非空操作，且没有用每幅图独立 min-max 归一化抹掉跨样本的绝对散射强度。

## 3. 模型结构

`GFFViTHorizonFormer` 包含四部分：

- ViT-B/16：224×224 六通道输入切成 14×14 个 patch token，利用全局自注意力学习易涝地形和 SAR 空间关系；首层用预训练 RGB 卷积核的通道均值扩展为六通道，避免随机丢弃全部预训练输入先验；
- U 形解码器：在 224/112/56 尺度建立卷积跳连，从 14×14 ViT token 逐级恢复到原分辨率，补偿纯 ViT 对细窄边界的不利影响；
- 时间分支：保留 20 天 ERA5、ERA5-Land、GloFAS 时空压缩与 Transformer，加入 24/48/72h horizon token；
- 多任务头：分割头、边界头、图块洪水存在性头联合训练。

正式模型共 89,564,939 个参数，其中冻结前 6 个 ViT block 后有 47,037,707 个可训练参数。ViT 可训练参数使用 `1e-5` 学习率，新建时序、跳连和解码模块使用 `1e-4`，降低预训练特征被小规模子集快速破坏的风险。

## 4. 实验控制

- 环境：现有 conda `daily`；未新建环境；
- 数据：与 MiT 基线完全相同的 GFF 固定 seed 1337 子集；
- 划分：fold 2–4 训练、fold 1 验证、fold 0 测试；
- 规模：3,000/500/500 个瓦片，每个瓦片展开 24/48/72h 三个样本；
- 因果约束：对目标日前最后 1/2/3 天再分析强迫置零，不读取 issue time 之后的信息；
- 类别与边界：50% 正瓦片均衡采样、Focal-BCE + Dice、5 像素边界监督与存在性辅助损失；
- 训练：batch 4、梯度累积 4（有效 batch 16）、AdamW、Cosine、4 轮；
- 模型选择：每一轮 checkpoint 都只在验证集上从 0.1–0.7 校准阈值，按验证宏 IoU 选轮次；测试集只在选定后评估。

## 5. Smoke 验证

48/24/24 瓦片、1 轮、无预训练权重的 smoke 实验已完整跑通训练、验证、阈值校准和测试，测试宏 IoU 0.0134、宏 Dice 0.0264。该数值只证明六通道数据、ViT 前后向和指标流水线可运行，不作为模型效果结论。

## 6. 正式结果

四轮固定阈值 0.3 的训练记录如下。第 2 轮固定阈值验证宏 IoU 最高，但第 3 轮的边界 F1 明显提高，说明不同任务头的收敛速度并不一致。

| 轮次 | Train loss | Val loss | 固定 0.3 Val 宏 IoU | Val 宏 Dice | 24/48/72h Boundary F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.3836 | 1.3808 | 0.0227 | 0.0443 | 0.0282/0.0274/0.0328 |
| 2 | 1.3497 | 1.3335 | **0.0363** | **0.0701** | 0.0051/0.0052/0.0073 |
| 3 | 1.3206 | 1.3745 | 0.0336 | 0.0651 | **0.0798/0.0806/0.0809** |
| 4 | 1.2953 | 1.3487 | 0.0363 | 0.0700 | 0.0537/0.0540/0.0554 |

### 6.1 逐轮验证集校准（主协议）

每轮都只在 validation 上选阈值后的排名为：第 3 轮 0.5/0.04369，第 1 轮 0.5/0.03781，第 4 轮 0.4/0.03743，第 2 轮 0.3/0.03634。因此不查看 test 的正式选择是第 3 轮、阈值 0.5。固定这一选择后，fold 0 test 结果如下。

| 提前量 | IoU | Dice | Precision | Recall | Accuracy | Boundary F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 24h | 0.0341 | 0.0660 | 0.1217 | 0.0453 | 0.9709 | 0.0059 |
| 48h | 0.0354 | 0.0685 | 0.1209 | 0.0477 | 0.9705 | 0.0066 |
| 72h | 0.0355 | 0.0686 | 0.1215 | 0.0478 | 0.9705 | 0.0069 |
| 宏平均 | **0.0350** | **0.0677** | 0.1214 | 0.0469 | 0.9706 | 0.0064 |

与旧 MiT 基线宏 IoU 0.0435、宏 Dice 0.0834 相比，校准主结果分别下降 19.5% 和 18.9%。因此本轮不能得出“ViT + CLAHE 已提高最终泛化性能”的结论。

### 6.2 预先固定阈值 0.3 的诊断结果

若严格沿用配置在训练前写定的阈值 0.3，并按该阈值的 validation 选择第 2 轮，则 test 的 24/48/72h IoU 为 0.0475/0.0479/0.0477，宏 IoU 0.0477、宏 Dice 0.0911，相对旧 MiT 基线提高 9.7%/9.2%。但逐轮校准协议在 validation 上选择了第 3 轮 0.5，而它在 test 上召回率只有约 4.7%。两组结果共同表明模型具有更好的候选表征，却存在明显的跨流域概率标定漂移；不能因为固定阈值结果更好，就在看过 test 后把它替换成主结果。

所有主协议数字来自 `outputs/gff_vit_sunet/run.json`，排名来自 `checkpoint_ranking.json`。固定阈值诊断来自训练完成时自动生成的第 2 轮初始测试结果。

### 6.3 可视化

- 默认第一个正样本：模型在阈值 0.5 下为空预测，TP/FP/FN 为 0/0/6,898，反映整体低召回问题；文件为 `prediction_72h.png`；
- 最大实测洪水样本也为空预测，TP/FP/FN 为 0/0/49,552；文件为 `prediction_72h_largest_flood.png`；
- 为观察模型在成功情况下学到的区域，量化评估结束后才对 500 个 72h test 样本做可视化排名。最佳案例 index 764 的 IoU 为 0.420，TP/FP/FN 为 2,049/2,716/115；文件为 `prediction_72h_best_case.png`。该案例明确标记为 best case，未参与模型、轮次或阈值选择。

最佳案例的五联图依次显示灾前 VV dB、CLAHE VV、实测 flooded 类、72h 预测概率与误差。CLAHE 的局部纹理更清楚，但预测仍明显覆盖过宽，与最佳案例中 2,716 个假阳性一致。

## 7. 复现

```powershell
conda run -n daily --no-capture-output python -m unittest tests.test_gff_pipeline -v
conda run -n daily --no-capture-output python -u -m src.train_gff --config configs\gff_vit_sunet_smoke.yaml
conda run -n daily --no-capture-output python -u -m src.train_gff --config configs\gff_vit_sunet.yaml
conda run -n daily --no-capture-output python -u scripts\select_gff_checkpoint.py --config configs\gff_vit_sunet.yaml --checkpoints outputs\gff_vit_sunet\epoch1.pt outputs\gff_vit_sunet\epoch2.pt outputs\gff_vit_sunet\epoch3.pt outputs\gff_vit_sunet\epoch4.pt
conda run -n daily --no-capture-output python scripts\predict_gff_example.py --config configs\gff_vit_sunet.yaml --checkpoint outputs\gff_vit_sunet\best.pt --horizon 3 --output outputs\gff_vit_sunet\prediction_72h.png
```

也可以执行 `powershell -ExecutionPolicy Bypass -File scripts\run_gff_vit_sunet_experiment.ps1` 复现完整闭环。

## 8. 结论边界

该实验比较的是固定小子集上的架构与输入处理，不等同于完整 98,853 个训练瓦片的最终能力。CLAHE 确实改善可视对比度，但不会生成新的观测信息；89.56M 参数的 ViT 相对 4.55M 参数 MiT 更容易在 3,000 瓦片上出现跨流域标定漂移。当前最合理的下一步不是用 test 回调阈值，而是扩大训练流域、在 validation 内做按事件/流域的概率校准，并分别消融“只换 dB+CLAHE”和“只换 ViT”，确认收益来源。业务化仍需要接入按历史 issue time 归档的 ECMWF/GloFAS forecast，并在完整数据和多随机种子上重新验证。
