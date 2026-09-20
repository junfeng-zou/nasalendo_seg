# ConvNeXt 人工头部框 A/B/C 实验

## 已完成结果

已使用RTX 3090 完成 3 组 × 3 个种子，共 9 次冻结主干的分类头训练。验证准确率均值：A=75.49%、B=84.31%、C=84.31%；macro-F1 均值：A=74.14%、B=82.05%、C=82.05%。B/C 的逐帧类别预测一致，但概率不同；尺度连接权重确实更新，替换尺度输入也会改变 C 的概率，本轮未产生额外类别预测增益。

[完整报告](../results/convnext_manual_head_abc_20260917/report.md) · [结果解读与尺度核查](../results/convnext_manual_head_abc_20260917/interpretation.md) · 逐图预测（本地实验产物，未随代码公开）。

5 项针对性测试通过。3 个验证类别 × 3 个方案的原图推理／缓存预处理／权重重载检查通过，最大类别概率误差为 2.98e-7；三组同种子初始化一致，源数据及标注指纹未改变。

## 目的

在同一批人工有框样本上，检查局部头部输入，以及裁剪前的显式尺度，是否改善距离状态分类。使用重新混合后的 `annotation_split`，不是原 CSV 的划分。

| 方案 | 视觉输入 | 额外尺度输入 |
|---|---|---|
| A_full | 原图等比例缩放、居中填充 | 两个固定零 |
| B_roi | 人工头部框自适应裁剪 | 两个固定零 |
| C_roi_scale | 与 B 完全相同的裁剪 | 框宽／FOV 直径、框高／FOV 直径 |

## 数据协议

- 训练 94 张：TooFar=31，Good=30，TooClose=33。
- 验证 34 张：TooFar=5，Good=15，TooClose=14。
- 50 张训练空标注、2 张验证空标注从三组共同排除，避免比较不同样本集合。
- 所有有框标注使用同一套坐标验证，支持 X-AnyLabeling 的两个对角点和四个顶点矩形；多框、非矩形、越界等不静默修复。
- 保留触及原图边缘的框并记录供复核，没有依赖验证成绩调整样本。人工范围正确性未被格式检查所保证。
- FOV 使用此前按每个视频图像光学边界拟合的固定直径，仅使用几何边界，不使用距离标签。文件及源数据指纹保存于 data_audit.json。
- 每幅图像先将头部框宽高扩到 1.2 倍，即每边增加原边长的 10%；越界填充黑色，完整裁剪后等比例缩放并居中填充到 384×384。原图输入同样等比例填充。
- C 的两个尺度特征来自扩框和缩放之前的人工框；均值、标准差只在训练集上计算。
- 三组不使用历史距离分类权重，避免重新混合后旧模型已见过部分当前验证图造成污染。

## 训练协议

ImageNet 预训练起点：本机 `convnext_tiny.in12k_ft_in1k` safetensors，严格加载后删除 1000 类分类层，保留原全局池化与归一化得到 768 维特征。主干完全冻结、eval 模式、FP32，不使用随机增强；先缓存特征，再训练分类头。

缓存训练与每个 batch 重新运行这一固定主干数学等价。B/C 共用同一份视觉特征缓存，区别仅在两个尺度通道。不是端到端微调，也不是之前的有序头／多层空间池化实验。

分类头：LayerNorm(768) → 拼接两个通道 → Linear(770,128) → GELU → Dropout(0.2) → Linear(128,3)。A/B 的尺度通道固定为零，C 使用标准化尺度；三组拥有相同参数结构、同种子相同初始化，A/B 的尺度连接因输入为零不参与学习。

普通三类交叉熵，无类别权重；AdamW，LR=3e-4，weight_decay=1e-3，batch=32，梯度裁剪 1.0。最多 100 轮，patience=20，以验证 macro-F1 选择最佳轮次，同分保留更早轮次。不同方案使用相同的种子和训练样本随机顺序。3 个预先指定种子为 42、43、44，逐图展示固定为 seed 42，不从结果挑选最佳种子。

概率直接取 argmax，无 0.55 接受阈值、时间平滑或机器人控制动作。

## 运行方式

在项目根目录运行：

```bash
PYTHON=python
$PYTHON distance_state_classifier/scripts/run_manual_head_abc.py --stage prepare
$PYTHON -m unittest discover -s tests -p 'test_manual_head_abc.py' -v
$PYTHON -u distance_state_classifier/scripts/run_manual_head_abc.py --stage train --device cuda
MPLCONFIGDIR=/tmp/nasalendo_abc_matplotlib $PYTHON distance_state_classifier/scripts/run_manual_head_abc.py --stage report
$PYTHON scripts/analyze_manual_head_abc.py
```

准备目录已存在时拒绝覆盖；已产生训练输出时拒绝再次训练。开始新实验应复制配置并更改 output，再通过 `--config` 指定。报告可以由已完成的预测和训练记录重新生成，不会重新训练。

## 输出

- [实验结果目录](../results/convnext_manual_head_abc_20260917)
- [配置](../distance_state_classifier/configs/manual_head_abc.json)
- `input_preview.html`：每张实际输入，左 A、右 B/C。
- `prepared.json` / `data_audit.json`：来源、划分、框、尺度、训练集标准化统计、源文件和输入指纹。
- `frozen_backbone.pt`：共用的冻结主干权重。
- `frozen_features.pt`：A 与 B/C 的特征及严格对应的 sample_id。
- `runs/{A_full,B_roi,C_roi_scale}/seed_{42,43,44}/best.pt`：分类头权重、方案、种子、最佳轮次、尺度归一化及 backbone 路径。
- `runs/.../history.json`：每轮训练与验证结果。
- `predictions.csv` / `prediction_gallery.html`：逐图结果，展示固定 seed 42。
- `aggregate.json` / `report.md` / `overview.png`：三组均值、样本标准差、各类召回与曲线。
- `inference_checks.json`：保存权重重新加载、原图在线预处理与缓存输入及概率的一致性核对。
- `scale_channel_checks.json` / `interpretation.md`：固定权重下的尺度通道核查及结果解释，不进行额外训练或调参。
- `completion.json`：只有全部训练和一致性检查成功后才生成。

## 结果边界

这是一轮混合视频划分的冻结特征预实验。34 张验证图参与选择最佳 epoch，不是独立测试集；三种子标准差不代表跨病例不确定性。TooFar 仅5张，一张对应召回率20个百分点。

结果只评估“已有人工头部框”的条件，不能直接推断自动检测误差、头部缺失帧或真实手术表现。新增尺度仍可能包含张合、姿态、遮挡及器械型号变化。后续是否训练头部检测器，需结合本轮结果和失败样本决定。
