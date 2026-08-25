# 洪析先知：GFF 未来洪水范围预测

当前主任务使用 Global Flood Forecasting（GFF）数据，在同一事件的目标时刻前 24/48/72 小时预测新增洪水范围。模型输入不含目标时刻 SAR：空间分支只读取灾前 Sentinel-1、DEM 和 HAND，时间分支读取 20 天 ERA5、ERA5-Land 与 GloFAS 强迫序列；目标是 GFF 标注中的 flooded 类（类别 2），永久水体不是正类。

需要注意：GFF 发布的是再分析数据，不是带 forecast issue time 的历史业务预报。正式配置使用严格因果模式：目标日前最后 1/2/3 天的再分析全部屏蔽，模型只读取 issue time 之前的数据，不会看到真实未来强迫。项目保留 `forcing_mode: perfect` 作为“把再分析当作零误差预报”的上界，但不把它当作主结果。实时上线仍应把被屏蔽时段替换为发布时刻可用的 ECMWF/GloFAS 预报并重新训练或微调。

## GFF 下载与训练

- 官方记录与说明：[GFF v3（Zenodo 14184289）](https://zenodo.org/records/14184289)
- 数据集 DOI：[10.5281/zenodo.14184289](https://doi.org/10.5281/zenodo.14184289)
- Zenodo API 清单：[record 14184289 API](https://zenodo.org/api/records/14184289)
- 数据论文配套代码：[Multihuntr/gff](https://github.com/Multihuntr/gff)

项目只下载当前模型使用的 `base + s1 + dem + hand + era5 + glofas`，不下载 HydroATLAS、WorldCover 和 extras。下载器支持多连接、断点续传、官方 MD5 校验和安全解压：

```powershell
conda run -n daily --no-capture-output python scripts\download_gff.py --root data\gff --components base glofas dem hand era5 s1 --workers 24 --part-mb 16
conda run -n daily --no-capture-output python -m src.train_gff --config configs\gff_horizonformer_smoke.yaml
conda run -n daily --no-capture-output python -m src.train_gff --config configs\gff_horizonformer.yaml
conda run -n daily --no-capture-output python scripts\select_gff_checkpoint.py --config configs\gff_horizonformer.yaml --checkpoints outputs\gff_horizonformer\epoch1.pt outputs\gff_horizonformer\epoch2.pt outputs\gff_horizonformer\epoch3.pt outputs\gff_horizonformer\epoch4.pt outputs\gff_horizonformer\epoch5.pt outputs\gff_horizonformer\epoch6.pt
conda run -n daily --no-capture-output python scripts\predict_gff_example.py --config configs\gff_horizonformer.yaml --checkpoint outputs\gff_horizonformer\best.pt --horizon 3 --output outputs\gff_horizonformer\prediction_72h.png
```

`GFFHorizonFormer` 由 MiT-B0 空间 Transformer、日尺度气象/水文时序 Transformer、horizon token、逐尺度 FiLM 融合和 SegFormer 解码器构成。训练阶段用 50% 正洪水瓦片重采样、Focal-BCE + Dice 解决类别不平衡，并用独立边界头约束细窄和不规则边缘。数据按官方流域 fold 划分：fold 0 测试、fold 1 验证、fold 2–4 训练，避免空间泄漏。

正式固定种子子集实验的最佳权重来自第 6 轮，验证集选择阈值 0.3。独立测试集 24/48/72 小时 IoU 分别为 0.0423/0.0442/0.0440，宏 IoU 0.0435、宏 Dice 0.0834。该结果是严格因果、3,000/500/500 瓦片的研究基线，尚不足以代表完整数据训练或业务可用性能。

完整任务定义、数据审计、结构说明和实验限制见 [`docs/GFF_HorizonFormer实验报告.md`](docs/GFF_HorizonFormer实验报告.md)。

## SU-Net 启发的 SAR 增强与 ViT 模型

新配置 `gff_vit_sunet.yaml` 参考 SU-Net 对 ISAR 图像保留原图并加入 CLAHE 增强图的做法，但针对 GFF 的 Sentinel-1 线性后向散射先做物理上合理的 dB 转换和固定范围裁剪。空间输入因此变为 `VV dB + VH dB + VV CLAHE + VH CLAHE + DEM + HAND` 六通道。模型使用 ImageNet 预训练 ViT-B/16 建模全局空间关系，并以卷积 U 形跳连恢复洪水边界；原有严格因果天气 Transformer、24/48/72h horizon 条件、类别均衡采样、Focal-Dice、边界头和存在性头全部保留。

SU-Net 原本用于非合作航天器 ISAR 位姿估计，本项目只迁移其双视图预处理思想，并不是复现 SU-Net 的任务或网络。依据见 [SU-Net 论文](https://www.nature.com/articles/s41598-023-38974-1) 与 [官方代码](https://github.com/Tombs98/SU-Net)。

```powershell
conda run -n daily --no-capture-output python -m src.train_gff --config configs\gff_vit_sunet_smoke.yaml
conda run -n daily --no-capture-output python -m src.train_gff --config configs\gff_vit_sunet.yaml
conda run -n daily --no-capture-output python scripts\select_gff_checkpoint.py --config configs\gff_vit_sunet.yaml --checkpoints outputs\gff_vit_sunet\epoch1.pt outputs\gff_vit_sunet\epoch2.pt outputs\gff_vit_sunet\epoch3.pt outputs\gff_vit_sunet\epoch4.pt
conda run -n daily --no-capture-output python scripts\predict_gff_example.py --config configs\gff_vit_sunet.yaml --checkpoint outputs\gff_vit_sunet\best.pt --horizon 3 --output outputs\gff_vit_sunet\prediction_72h.png
```

只运行这一版完整闭环也可使用 `scripts\run_gff_vit_sunet_experiment.ps1`。正式对照仍使用相同的 seed 1337、严格因果输入和 3,000/500/500 瓦片，避免把输入规模变化误认为架构收益。

正式四轮统一验证校准选择第 3 轮、阈值 0.5，独立测试宏 IoU 0.0350、宏 Dice 0.0677，低于 MiT 基线的 0.0435/0.0834。预先固定阈值 0.3 的第 2 轮测试可达到 0.0477/0.0911，但不能在查看 test 后改用这组较好数字作为主结果；差异说明小子集 ViT 存在明显跨流域概率标定漂移。完整结果、失败案例和最佳案例见 [`docs/GFF_ViT_SU-Net实验报告.md`](docs/GFF_ViT_SU-Net实验报告.md)。

已完成 Standard、仅 dB、仅 CLAHE、dB + CLAHE 双视图四组受控消融。每组均联合扫描 4 个 epoch 与验证阈值，并且 test 只用于最终报告。验证集最终选择 **仅 dB**：validation/test Macro IoU 为 **0.0443/0.0508**，test Macro Dice 为 **0.0967**；双视图对应 0.0437/0.0350/0.0677。当前证据表明收益主要来自物理 dB 表达，CLAHE 双视图在小样本预算下没有额外泛化收益。完整表格、选择偏差说明和对比图见 [`docs/GFF_SU-Net预处理消融实验报告.md`](docs/GFF_SU-Net预处理消融实验报告.md)。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gff_sunet_ablation.ps1
```

消融胜出的 dB-only 模型随后从最佳检查点出发，在 fold 2–4 的全部 98,853 个训练瓦片上完成 2 个 epoch 微调；验证/测试各固定 2,000 个瓦片。验证集联合选择第 2 轮、阈值 0.3，测试 24/48/72h IoU 为 **0.0631/0.0633/0.0633**，Macro IoU/Dice 为 **0.0632/0.1189**。该实验的评估子集大于消融实验，不能把两组数值当作严格配对增益。配置和一键脚本分别为 `configs/gff_vit_db_full_finetune.yaml` 与 `scripts/run_gff_vit_db_full_finetune.ps1`。

已增加逐热度图有效像素 min–max 归一化后再阈值化的可选后处理，validation 重新选择第 2 轮、阈值 0.75。其 validation/test Macro IoU 为 0.0492/0.0918，测试 Macro Dice 为 0.1682；同一默认 72h 漏检样例的 IoU 从 0 提高到 0.8102。由于 validation 反而低于原始概率方案，且测试边界 F1 从 0.0641 降至 0.0194，该方案只作为概率尺度漂移诊断保留，不能根据已经查看的 test 结果替换正式主结果。配置与结果见 `configs/gff_vit_db_full_finetune_per_heatmap.yaml`、`outputs/gff_vit_db_full_finetune_per_heatmap/run.json` 和 [`docs/GFF_SU-Net预处理消融实验报告.md`](docs/GFF_SU-Net预处理消融实验报告.md)。

进一步实现了仅对低置信度热度图归一化的自适应方案：以有效像素 `q99` 判断整体置信度，validation 选择分流门限 0.01；约 28.43% 的 test 图使用 min–max + 0.75，其余直接使用原始概率 + 0.30。该方案 validation/test Macro IoU 为 0.0558/0.0682，Test Dice 0.1276，边界 F1 0.0641。它缓解了全部归一化造成的边界退化，但 validation 仍未超过全部原始方案的 0.0588，故保留为可选后处理。配置为 `configs/gff_vit_db_full_finetune_adaptive.yaml`。

也可一键执行下载校验、数据审计、测试、smoke、正式子集训练和预测示例：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gff_experiment.ps1
```

## 旧版 ImpactMesh 灾中制图实验

仓库仍保留旧实验的代码、权重结果和报告，原始 ImpactMesh 数据已从工作目录移除。以下章节记录历史灾中制图结果，不代表当前未来预测任务。

原阶段一实现了 ImpactMesh-Flood 数据核验、E0-E3 输入消融、U-Net 二分类分割、IoU/Dice/Precision/Recall/F1 评估、checkpoint 与可视化归档。

## 快速开始

```powershell
conda run -n daily python -m pip install -r requirements.txt
conda run -n daily python scripts\download_data.py --split val --modalities MASK DEM S1RTC
conda run -n daily python scripts\inspect_data.py --data-root data\raw --split val --limit 20
conda run -n daily python -m src.train --config configs\e0_smoke.yaml
```

正式消融实验使用 `configs/formal_e0.yaml`、`formal_e1.yaml`、`formal_e2.yaml`、`formal_e3.yaml`；`configs/e0.yaml` 至 `e3.yaml` 保留为更大训练预算的扩展配置。数据归档来自官方 Hugging Face 仓库，解包后目录应为 `data/raw/<split>/<modality>/*.zarr.zip`。

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

## 模型结构改进

`siamese_change_unet` 将 E2 的灾前、灾中双极化 SAR 分别送入共享编码器，并在四个尺度融合灾中语义特征与绝对时相差异，再由 U-Net 解码器恢复洪水边界。它与上一节的类别权重和边界带损失组合使用。

固定 seed 42 的正式实验中，validation IoU 为 0.5968；562 个未见事件 holdout 图块上的 IoU/Dice/Boundary F1 为 0.5342/0.6964/0.5524，原 E2 对应为 0.4941/0.6614/0.5329。当前未继续运行其他随机种子，结果应视为单种子结构验证。

```powershell
conda run -n daily python -m src.train --config configs\formal_e2_siamese_change.yaml
conda run -n daily python scripts\evaluate_checkpoint.py --config configs\formal_e2_siamese_change.yaml --split test --sample-list data\split\impactmesh_flood_test_holdout.txt --name test_holdout
```

设计与完整对照见 `docs/模型结构改进实验报告.md`，机器可读结果见 `docs/model_structure_results.csv`。

## 轻量分割模型与 SpaceNet 8 参考实验

项目已在不增加 Python 依赖的前提下接入 `deeplabv3plus_mobilenet` 和原生 `segformer_b0`，并保持 seed 42、完整 train/val、12 epoch 与同一损失设置。单模型 holdout IoU 分别为 0.5130 和 0.5268，均未超过 `siamese_change_unet` 的 0.5342；SegFormer-B0 的 Boundary F1 0.5542 为单模型最高。

按 SpaceNet 8 获奖方案的异构集成思路，仅在 validation 选择组合后评估一次 holdout。在该轮实验中，等权 `SiameseChangeUNet + DeepLabV3+` 的 holdout IoU 为 0.5409、Dice 为 0.7020，但正事件宏 IoU 和推理效率仍以单 Siamese 更优。

```powershell
conda run -n daily python -m src.train --config configs\formal_e2_deeplabv3plus.yaml
conda run -n daily python -m src.train --config configs\formal_e2_segformer_b0.yaml
conda run -n daily python scripts\evaluate_ensemble.py --configs configs\formal_e2_siamese_change.yaml configs\formal_e2_deeplabv3plus.yaml --split test --sample-list data\split\impactmesh_flood_test_holdout.txt --name test_holdout --output-dir outputs\ensemble_siamese_deeplab
```

完整分析见 `docs/轻量分割模型与SpaceNet8参考实验报告.md`。

## 洪水存在性辅助头与时相 Transformer

为降低无洪水图块误报，`siamese_change_unet_presence` 在 SiameseChangeUNet 瓶颈上增加图像级存在性分类头。训练时联合优化像素分割损失与图块级 BCE，推理时使用存在概率对分割概率作软门控。固定 seed 42 的正式实验取得 validation IoU 0.6114，holdout IoU/Dice/Boundary F1 为 **0.5540/0.7130/0.6016**，正洪水事件宏 IoU 为 0.3794；无洪水事件误报像素由原 E2 的 1,599 降至 1,510。

同时测试了共享 MiT-B0 编码器、逐尺度时相差异融合的 `siamese_segformer_b0`。其 holdout IoU 为 0.4860，未超过普通 SegFormer-B0 和 SiameseChangeUNet，因此保留为对照而不作为主模型。validation 选出的 `presence + SiameseChangeUNet` 等权集成在 holdout 上也未超过存在性单模型，最终主候选保持为 `siamese_change_unet_presence`。

```powershell
conda run -n daily python -m src.train --config configs\formal_e2_siamese_presence.yaml
conda run -n daily python scripts\evaluate_checkpoint.py --config configs\formal_e2_siamese_presence.yaml --split test --sample-list data\split\impactmesh_flood_test_holdout.txt --name test_holdout

conda run -n daily python -m src.train --config configs\formal_e2_siamese_segformer.yaml
conda run -n daily python scripts\evaluate_checkpoint.py --config configs\formal_e2_siamese_segformer.yaml --split test --sample-list data\split\impactmesh_flood_test_holdout.txt --name test_holdout
```

设计、分类头诊断、集成对照与限制见 `docs/洪水存在性辅助头实验报告.md`。

## 模型架构二次优化

`siamese_change_unet_presence_refine` 在存在性模型上进一步加入三项结构：瓶颈处的轻量多尺度空洞上下文、由粗尺度解码特征控制的跳连注意力，以及带独立边界监督的边缘细化模块。它保留原有图块存在性软门控。

同样使用 seed 42 和 12 epochs，新模型的 validation IoU 为 0.6165；holdout IoU/Dice/Precision 为 **0.5594/0.7175/0.7041**，均为当前最高，正洪水事件宏 IoU 也提高到 0.3854。Boundary F1 为 0.5925，低于原 Presence 模型的 0.6016；无洪水事件误报为 1,775，亦高于原 Presence 的 1,510。因此按主要区域 IoU 选择新模型，边界质量或低误报优先时仍保留原 Presence 模型。

```powershell
conda run -n daily python -m src.train --config configs\formal_e2_siamese_presence_refine.yaml
conda run -n daily python scripts\evaluate_checkpoint.py --config configs\formal_e2_siamese_presence_refine.yaml --split test --sample-list data\split\impactmesh_flood_test_holdout.txt --name test_holdout
conda run -n daily python scripts\benchmark_checkpoint.py --configs configs\formal_e2_siamese_presence.yaml configs\formal_e2_siamese_presence_refine.yaml
```

完整结构、逐事件对照和效率分析见 `docs/模型架构二次优化实验报告.md`。

## 下一次洪水预测概念验证

现有 E2 模型属于灾中制图，不是未来预报。为验证“使用同一地点上一次洪灾的 pre-event/event SAR，预测下一次洪灾范围”，项目新增跨事件空间拼接数据管线和严格时间留出实验。模型输入中不包含下一次洪灾的 event 影像；目标才是下一次洪灾的真实掩膜。

本地 24,578 个对齐图块中，经同一 MGRS 区域历史影像拼接和至少 95% 覆盖过滤，仅得到 458 个有效下一事件样本，按目标年份划分为 train 327（不晚于 2023）、val 35（2024）、test 96（2025–2026）。方向时相融合模型的测试 IoU 为 0.3885；预训练历史制图加未来残差为 0.4092。二者均低于“识别上一次洪水并延续到下一次”的验证集选阈值基线 0.4470。保守全模型微调的最佳权重停留在 epoch 0，也没有学到有效的未来修正。

因此当前结果只能作为洪水复发空间风险概念验证，不能称为可部署的下一次洪水预报。ImpactMesh-Flood 官方定位也是 flood mapping；缺少下一次事件的降雨预报、河流水位/流量、土壤湿度和明确预见期，无法唯一确定未来淹没范围。

```powershell
conda run -n daily python scripts\audit_future_prediction_data.py
conda run -n daily python scripts\build_future_event_manifest.py
conda run -n daily python -m src.train --config configs\formal_future_event_forecast.yaml
conda run -n daily python scripts\evaluate_checkpoint.py --config configs\formal_future_event_forecast.yaml --split test --name future_test
conda run -n daily python scripts\evaluate_future_baselines.py --forecast-config configs\formal_future_event_forecast.yaml --mapping-config configs\formal_e2_siamese_presence_refine.yaml --partition test
```

完整审计、模型设计、失败对照和下一步数据要求见 `docs/下一次洪水预测可行性与实验报告.md`。
