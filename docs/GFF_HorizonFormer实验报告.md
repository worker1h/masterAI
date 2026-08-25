# GFF 24/48/72 小时洪水范围预测实验报告

## 1. 任务修正与结论边界

此前 ImpactMesh-Flood 实验输入了灾前和灾中 SAR，输出本质上是灾中洪水制图，不是未来预测。本轮改用 Global Flood Forecasting（GFF）v3，在目标洪水时刻前 24、48、72 小时分别预测目标时刻的 flooded 类范围。目标时刻 Sentinel-1 只用于数据集制图标签生成，不进入模型。

GFF 中气象和水文变量是 ERA5、ERA5-Land 与 GloFAS 再分析，并非按历史业务发布时间归档的预报。为杜绝未来信息泄漏，正式配置使用 causal hindcast：对 24/48/72 小时任务分别把 issue time 之后的 1/2/3 天再分析置为归一化均值，模型只能读取发布时刻之前的观测历史。`forcing_mode: perfect` 可把这些时段保留为“零误差预报”上界，但不作为主结果。业务部署应把被屏蔽时段换成 issue time 当时真正可用的 ECMWF/GloFAS 预报并重新训练或微调。

## 2. 数据来源与清理

- 官方下载页：https://zenodo.org/records/14184289
- 官方文件清单 API：https://zenodo.org/api/records/14184289
- 官方代码：https://github.com/Multihuntr/gff

本模型实际使用 `base.zip`、`glofas.zip`、`dem.zip`、`hand.zip`、`era5.zip` 和 `s1.zip`。HydroATLAS、WorldCover 和 extras 没有进入当前网络，未继续下载；旧 ImpactMesh 原始数据已移入 Windows 回收站，历史代码、checkpoint、划分文件和报告均保留。

`base.zip` 含 298 个 ERA5 覆盖事件元数据，其中官方 floodmap partition 覆盖 295 个事件、150,321 个 224×224 瓦片。按 Level-4 流域 fold 做空间隔离：fold 0 为测试、fold 1 为验证、fold 2–4 为训练。可用规模如下。

| 划分 | 事件数 | 瓦片数 | 含洪水瓦片 | flooded 像素占比 |
|---|---:|---:|---:|---:|
| train（fold 2–4） | 190 | 98,853 | 22,478（22.74%） | 2.02% |
| val（fold 1） | 33 | 19,729 | 4,233（21.46%） | 1.18% |
| test（fold 0） | 72 | 31,739 | 6,950（21.90%） | 2.56% |

可见不平衡同时存在于瓦片级和像素级：训练集中约 77% 瓦片没有 flooded 像素，即便在全部有效像素中 flooded 也只有约 2%。

## 3. 输入、标签与预见期

每个样本包含：

1. 空间分支：灾前 Sentinel-1 VV/VH、DEM、HAND，共 4 通道，10 m 分辨率；
2. 时间分支：截止目标日的 20 天 ERA5 9 变量、ERA5-Land 14 变量、GloFAS 3 变量，共 26 通道日序列；每个本地瓦片读取周边 50 km 水文气象上下文并重采样为 16×16，避免把粗分辨率强迫退化为单像素常数；
3. 预见期：1/2/3 天 horizon token，并用二值序列标出 issue time 之后不可见的时段；因果配置把这些时段置为归一化均值，确保目标日前真实再分析不会泄漏；
4. 标签：GFF floodmap 中类别 2（flooded）。类别 1 永久水体是负类，类别 3 无效区域不计损失和指标。

“新增洪水”在本报告中指相对于正常背景和永久水体新增的临时淹没区域，不是把目标时刻影像与灾前影像直接相减得到的伪预测。

## 4. 模型结构

主模型 `GFFHorizonFormer` 是面向多源、异分辨率数据的双分支 Transformer：

- 空间编码器使用 dependency-free MiT-B0/SegFormer 四尺度结构，提取 SAR 与地形的局部易涝特征；
- 每天气象/水文栅格先经轻量 CNN 压缩成 token，再由 2–3 层时间 Transformer 建模 20 天累积过程；
- horizon token 区分 24/48/72 小时任务，observed/forecast embedding 区分已观测与预报强迫；
- 时间上下文通过逐尺度 FiLM 注入四级空间特征，再由 SegFormer 解码器恢复 224×224 洪水概率；
- 附加图块洪水存在性头和边界头，分别约束误报与不规则边缘。

## 5. 类别不平衡与边界处理

- 训练采样器把正洪水瓦片目标比例提高到 50%，但验证和测试保持原始分布；
- 像素损失使用带正类权重的 Focal-BCE 与 soft Dice，兼顾难例和区域重叠；
- 由标签的形态学梯度生成 5 像素边界带，独立边界头使用加权 BCE + Dice；
- 无效像素在全部分割、边界损失和指标中显式屏蔽。
- 最终概率阈值只在 validation 的 0.1–0.7 候选中按三预见期宏 IoU 选择，然后固定一次评估 test，避免用测试集调阈值。

## 6. 实验配置

运行环境为现有 conda `daily`，Python 3.14、PyTorch 2.11、CUDA 13.0，显卡 RTX 5070 12 GB；没有创建新环境。正式模型为 4.55M 参数，实测 batch size 16 的一次 224×224 前向与反向峰值显存约 1.1 GiB。正式配置使用 batch size 16、AdamW、初始学习率 2e-4、Cosine 调度和混合精度。为在单卡上完成可复现实验，正式配置限定为 3,000/500/500 个 train/val/test 瓦片，每个瓦片展开三个预见期；该结果必须标记为固定种子的子集实验，不等同于完整 98,853 瓦片训练。

## 7. 实验结果

正式训练完成后对第 1、3、4、5、6 轮 checkpoint 使用相同 validation 和候选阈值做统一校准。第 6 轮以 validation 宏 IoU 0.0387 胜出，阈值为 0.3；随后固定模型与阈值，在从未参与训练或调参的 fold 0 test 上得到下表。全部数字来自 `outputs/gff_horizonformer/run.json`，不是推测值。

| 预测提前量 | IoU | Dice | Precision | Recall | Accuracy | Boundary F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 24 小时 | 0.0423 | 0.0812 | 0.0514 | 0.1939 | 0.9003 | 0.0261 |
| 48 小时 | 0.0442 | 0.0847 | 0.0540 | 0.1971 | 0.9032 | 0.0340 |
| 72 小时 | 0.0440 | 0.0843 | 0.0537 | 0.1964 | 0.9030 | 0.0347 |
| 宏平均 | **0.0435** | **0.0834** | 0.0530 | 0.1958 | 0.9022 | 0.0316 |

结果表明模型已学到高于空预测的空间信号，但 Precision、Recall、IoU 仍低，尚不具备业务部署性能。约 0.90 的 Accuracy 主要由占绝大多数的非洪水像素贡献，不能单独作为效果依据。严格屏蔽未来再分析、训练子集仅 3,000 个瓦片、没有历史业务预报强迫，是当前上限较低的主要原因。

72 小时可视化样本 `4711-7080012950-2018-11-23-2018-11-29-2018-12-05` 的 IoU 为 0.252：真阳性 6,756 像素、假阳性 19,912、假阴性 142。模型覆盖了主要目标区域，但预测面积明显偏大，与全测试集低 Precision 的结论一致。图像保存于 `outputs/gff_horizonformer/prediction_72h.png`。

## 8. 复现命令

```powershell
conda run -n daily --no-capture-output python scripts\download_gff.py --root data\gff --components base glofas dem hand era5 s1 --workers 24 --part-mb 16
conda run -n daily --no-capture-output python scripts\audit_gff.py --root data\gff --output outputs\gff_data_audit.json
conda run -n daily --no-capture-output python -m unittest discover -s tests -v
conda run -n daily --no-capture-output python -m src.train_gff --config configs\gff_horizonformer_smoke.yaml
conda run -n daily --no-capture-output python -m src.train_gff --config configs\gff_horizonformer.yaml
conda run -n daily --no-capture-output python scripts\select_gff_checkpoint.py --config configs\gff_horizonformer.yaml --checkpoints outputs\gff_horizonformer\epoch1.pt outputs\gff_horizonformer\epoch2.pt outputs\gff_horizonformer\epoch3.pt outputs\gff_horizonformer\epoch4.pt outputs\gff_horizonformer\epoch5.pt outputs\gff_horizonformer\epoch6.pt
conda run -n daily --no-capture-output python scripts\predict_gff_example.py --config configs\gff_horizonformer.yaml --checkpoint outputs\gff_horizonformer\best.pt --horizon 3 --output outputs\gff_horizonformer\prediction_72h.png
```

## 9. 限制与下一步

1. 因果配置没有使用未来天气预报，可能低估可用预报带来的收益；需要用 archived ECMWF/GloFAS forecast rerun 填充屏蔽时段，做严格的 issue-time 回测；
2. 当前固定 24/48/72 小时标签是从事件目标日构造的回报预测视角，GFF 原始标签没有三套独立预见期观测；
3. 当前单种子、固定子集实验只验证闭环，后续应扩展到全部训练流域和多个随机种子；
4. 可进一步增加 HydroATLAS/WorldCover 静态先验，但只有在消融证明有效后才应下载并纳入提交数据清单。
