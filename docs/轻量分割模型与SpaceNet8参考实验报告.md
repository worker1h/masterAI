# 轻量分割模型与 SpaceNet 8 参考实验报告

日期：2026-08-14

环境：既有 conda `daily` 环境，未创建新环境、未安装额外模型库

数据：ImpactMesh-Flood E2（pre-event + event VV/VH SAR）

实验协议：固定 seed 42，完整 train/val，12 epoch，`pos_weight=2`，BCE + Dice + 边界带损失

## 1. 接入模型

### DeepLabV3+-MobileNetV3

使用 MobileNetV3-Small 编码器、ASPP 多尺度空洞卷积和低层特征解码器，输入层改为四通道。该实现保留 DeepLabV3+ 的多尺度上下文与浅层细节融合思想，未使用光学 RGB 预训练权重。DeepLabV3+ 原始设计见 [ECCV 2018 论文](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.php)。

### SegFormer-B0

原生实现四阶段 MiT-B0：重叠 patch embedding、高效空间缩减注意力、Mix-FFN 和轻量多尺度解码器。训练使用 PyTorch 融合 `scaled_dot_product_attention`，避免新增 transformers、timm 或 mmsegmentation 依赖。结构依据 [SegFormer 官方实现](https://github.com/NVlabs/SegFormer)与 [NeurIPS 论文](https://papers.nips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)。

## 2. 正式结果

holdout 包含 562 个图块、12 个训练未见事件；指标阈值固定为 0.5，Boundary F1 容差为 2 像素。

| 模型 | Val IoU | Holdout IoU | Dice | Precision | Recall | Boundary F1 | 正事件宏 IoU | 空事件误报像素 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 类别/边界 U-Net | 0.5757 | 0.4981 | 0.6650 | 0.5955 | 0.7528 | 0.5420 | 0.3334 | 2,155 |
| **SiameseChangeUNet** | 0.5968 | **0.5342** | **0.6964** | 0.6291 | 0.7797 | 0.5524 | **0.3658** | **2,219** |
| DeepLabV3+-MobileNetV3 | 0.5714 | 0.5130 | 0.6781 | **0.6406** | 0.7202 | 0.5174 | 0.3267 | 4,395 |
| SegFormer-B0 | 0.5622 | 0.5268 | 0.6901 | 0.6098 | **0.7947** | **0.5542** | 0.3570 | 3,020 |

结论：

- DeepLabV3+ 更偏 Precision，但 Recall 和边界明显下降；它适合作为互补成员，不适合替换主模型。
- SegFormer-B0 的全局 IoU 略低于 SiameseChangeUNet，但 Boundary F1 和 Recall 更高，说明全局上下文对连续洪水区域有效。
- 单模型综合表现仍以 SiameseChangeUNet 最好。显式双时相共享编码和差异融合，比直接把四通道送入通用分割骨干更适合当前 SAR 任务。

## 3. SpaceNet 8 前五方案能否参考

可以参考方法设计，但不能直接搬用权重或分数。SpaceNet 8 是高分辨率光学影像上的道路、建筑及其洪水属性识别；当前任务是 SAR 上的新增洪水像素分割，传感器、分辨率和标签定义不同。

[SpaceNet 8 官方获奖代码仓库](https://github.com/SpaceNetChallenge/SpaceNet8)列出的前五方案中，第 1 名使用 Siamese HRNet+OCR，第 2/4 名使用 Siamese ResNet50 U-Net，第 5 名集成多个 Siamese EfficientNet U-Net，第 3 名的洪水分支使用 Siamese Swin+UPerNet。官方的[获奖方案复盘](https://medium.com/@SpaceNet_Project/spacenet-8-a-closer-look-at-the-winning-approaches-75ff4033bf53)还显示，前列方案普遍使用额外数据、数据清洗、增强、辅助任务、阈值/形态学后处理和模型集成。

| 名次 | 关键方法 | 对当前项目的可迁移价值 |
|---|---|---|
| 1 | Siamese HRNet+OCR、RMI、保守洪水阈值、孤立误报抑制 | 保留双时相结构；在 validation 校准阈值并抑制孤立预测 |
| 2 | Siamese U-Net、图像级洪水分类辅助损失、Focal+Dice | **最高优先级**：增加“图块是否有洪水”分支，解决空场景误报 |
| 3 | Siamese Swin+UPerNet、SegFormer、Lovasz、强增强 | Transformer 可作为互补模型；不应假定比显式变化检测更强 |
| 4 | Siamese ResNet50、xBD 等额外数据 | 可研究灾害数据预训练，但必须处理光学到 SAR 的域差异 |
| 5 | EfficientNet 集成、边缘/接触/路口辅助任务、清洗错标 | 借鉴边缘辅助头、数据质量审计和异构集成 |

## 4. 等权集成验证

为避免 holdout 调参，先在 validation 上比较三组固定等权组合：

| 组合 | Val IoU | Val Boundary F1 |
|---|---:|---:|
| Siamese + SegFormer | 0.5966 | **0.5874** |
| **Siamese + DeepLabV3+** | **0.6007** | 0.5517 |
| 三模型 | 0.5995 | 0.5685 |

只将 validation IoU 最高的 Siamese + DeepLabV3+ 评估一次 holdout：

| Holdout IoU | Dice | Precision | Recall | Boundary F1 | 正事件宏 IoU | 空事件误报像素 |
|---:|---:|---:|---:|---:|---:|---:|
| **0.5409** | **0.7020** | **0.6580** | 0.7524 | 0.5514 | 0.3578 | 2,908 |

该集成取得当前最高的全局 IoU 和 Dice，但正事件宏 IoU 仍低于单 Siamese，说明收益偏向像素较多的大事件。若竞赛主指标是全局像素 IoU，可把它作为提交候选；若强调事件间均衡泛化或低时延，仍应使用单 Siamese。

## 5. 计算开销

RTX 5070、batch size 1、4×256×256 输入、15 次预热和 60 次计时：

| 模型 | 参数量 | 单图中位延迟 |
|---|---:|---:|
| SiameseChangeUNet | 526,593 | 1.38 ms |
| DeepLabV3+-MobileNetV3 | 3,472,289 | 3.17 ms |
| SegFormer-B0 | 3,806,337 | 3.46 ms |
| Siamese + DeepLabV3+ | 3,998,882 | 4.70 ms |

## 6. 下一步建议

不建议继续堆叠更大的通用分割骨干。优先顺序为：

1. 在 SiameseChangeUNet 上加入图像级洪水存在性辅助头，并用分类概率门控空场景输出。
2. 只在 validation 上校准分割阈值和最小连通域面积，针对当前空事件误报。
3. 增加稳定水体抑制或永久水体辅助任务，减少河道暗散射误判。
4. 若提交推理预算允许，保留 Siamese + DeepLabV3+ 等权集成；SegFormer 更适合作为边界/召回型消融或后续蒸馏教师。

本轮遵循要求，没有运行其他随机种子。完整机器可读结果见 `docs/model_structure_results.csv`，权重与逐事件指标位于本地 `outputs/formal_e2_deeplabv3plus/`、`outputs/formal_e2_segformer_b0/` 和 `outputs/ensemble_siamese_deeplab/`。
