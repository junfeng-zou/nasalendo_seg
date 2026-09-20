# 实验 B：保留尺度的外观增强与预测一致性

本轮已于 2026-09-07 完成正式训练与 A/B 干预评估：40 轮触发早停，最佳权重来自第 25 轮。测试 accuracy 从 A 的 85.25% 降至 B 的 67.21%；颜色干预平均翻转率略降，但平均概率 TV 未改善。本轮结果不支持用 B 替换 A。

详见 实验 B 结果报告（本地实验产物，未随代码公开） 和 逐图干预展示（本地实验产物，未随代码公开）。9 项新增训练/增强测试、6 项既有诊断测试通过；完成了真实 EndoDAC 的 GPU 训练与严格权重加载验证。

## 实验问题

在不增加真实手术训练数据的前提下，降低距离状态分类器对假模颜色、器械颜色的依赖，同时检查原图上的分类能力是否保留。

本次对象是已有的 `endodac_encoder_classifier`，对应实验 A 的权重：

```text
distance_state_classifier_endodac/runs/endodac_encoder_multiscale_adapter_reweight_soft_392/best.pt
```

它是 `multiscale_mean` 编码器分类模型。B 没有引入 mask-aware 分类结构；mask 只用于训练期间选择增强区域，推理仍只需要 RGB。原有实时控制脚本不需要修改。

## 固定条件与变化

| 项目 | 实验 B |
|---|---|
| 数据划分 | 沿用 A：956 张训练、175 张验证、122 张测试，无视频交叉 |
| 类别 | `TooFar / Good / TooClose` |
| 分类输入 / 编码器输入 | 392×392 / 224×280，与 A 相同 |
| 编码器与分类头 | EndoDAC base、四层特征均值拼接、原分类头 |
| 可训练部分 | 前 5 轮分类头，之后 adapter 与分类头 |
| 训练上限 / 早停 | 60 轮 / 验证集 macro-F1 连续 15 轮不提高 |
| batch size | 8 个原始样本，每个样本生成 2 个视图，总前向图像数为 16 |
| 学习率、类别权重、随机种子 | 从 A 的 checkpoint 配置复制，不另行调整 |
| 初始化 | 配置指定的预训练 EndoDAC + 新分类头，不从 A 的分类权重继续训练 |
| 变化 | 取消旧训练增强，改为保留尺度的外观增强；增加双视图监督和 JS 一致性损失 |
| 最佳模型选择 | 仅使用未经干预的验证集 macro-F1 |

启动时会把 `data/image/model/train/inference` 五个配置节逐项与 A 的 checkpoint 比对，不一致时直接报错。配置一致不等同于证明历史训练过程完全可重现：旧 checkpoint 没有记录当时是否额外传入 `--init-checkpoint`，本次初始化方式已明确写入审计记录。

A 的旧增强中包含 `scale_frac: 0.08`、旋转和位移。B 取消这些操作，所以 A/B 比较的是这一整套训练改动的效果，不能单独归因于 JS 损失。如果需要区分增强与一致性的贡献，后续应增加同样增强但 `weight: 0` 的消融。

## 增强如何保持尺度

先按照 A 的方式把图像缩放到 392×392，然后在这张画布上生成增强图。原图视图与 A 的预处理逐元素一致，已用三个类别的实际样本检查。

- 分别随机调整器械内部与背景的 Lab 色度：旋转范围 ±35°、色度幅值 0.5–1.4 倍；选中的区域有 20% 概率去色。
- 背景、器械区域的选择概率分别为 0.85、0.70；另有 10% 概率保留原图。
- 30% 概率加入轻微明度变化，Lab L 偏移最多 ±3，明度对比度 0.95–1.05。
- 保留器械轮廓两侧的 2 像素保护带，以及有效视野边缘和外部黑边。
- 不增加几何缩放、随机裁切、形变、遮挡、模糊或噪声。两视图共享原始类别标签。

色度变换过程中保持 Lab L；转回 RGB 时，色域裁剪和量化可能带来少量亮度变化。因此不能把它描述为严格物理意义上的“只换颜色”。保护带中的原始像素严格不变。

mask 从已有缓存读取，缺失、损坏或尺寸不符会报错。空 mask 样本保留原图，避免把未知器械误当作背景；器械内部保护带腐蚀后为空，则只进行可用背景增强。此次全量检查：无缺失或空 mask，1 张图的器械内部区域过细而不可增强，未删除样本。

预览覆盖训练集每个视频/标签组合各一张，每张提供 3 个确定随机种子的增强结果。必须结合预览理解增强强度：这不会生成真实手术中的血液、烟雾、组织形变或新的器械几何，也不能解决所有假模到真实手术的差异。

## 损失与训练记录

设原图和增强图预测为 \(p, q\)，标签为 \(y\)：

\[
L=\frac12\left[CE(p,y)+CE(q,y)\right]+\lambda JS(p,q),\qquad \lambda=1.
\]

CE 沿用 A 的类别权重：TooFar=0.45、Good=1.15、TooClose=1.35。JS 使用自然对数、对称计算，对两个视图都反向传播，计算使用 float32。训练时保留原模型的 dropout，所以训练 JS 也包含随机网络扰动的影响；离线干预评估使用 `eval()`。

记录每轮 CE、JS、原图训练指标、验证混淆矩阵、学习率、实际参数更新数、AMP 跳过更新数和 GradScaler 比例。AMP 初始动态缩放可能跳过个别溢出步骤；整轮没有完成任何参数更新会报错。checkpoint 包含优化器、调度器、GradScaler、历史记录和随机状态，可在 epoch 边界恢复。

## 运行

在仓库根目录、具有 GPU 访问权限的终端中运行：

```bash
PYTHON=python
$PYTHON -u distance_state_classifier_endodac/scripts/train_appearance_consistency.py --device cuda
```

配置文件：

```text
distance_state_classifier_endodac/configs/distance_state_endodac_experiment_b.yaml
```

默认输出目录：

```text
distance_state_classifier_endodac/runs/endodac_experiment_b_appearance_consistency_20260907/
```

输出内容：

- `data_audit.json`：划分、类别数量、视频名、CSV/预训练权重指纹、mask 检查。
- `resolved_config.json`、`runtime.json`：实际配置、设备和本次训练代码指纹。
- `augmentation_preview.html`、`preview/`：训练增强预览。
- `best.pt`、`last.pt`、每 10 轮 checkpoint：独立保存的 B 权重。
- `metrics.json`：每轮训练和验证记录。
- `test_metrics.json`、`test_confusion_matrix.csv`：训练结束后才生成。
- `completion.json`：正式训练完成标志。

输出目录已有权重时拒绝覆盖。训练过程中有 `.training.lock` 防止同一目录并发写入。如果进程被强制终止，锁可能残留；确认原进程已结束后才可移除该锁。

仅检查数据和生成预览：

```bash
$PYTHON -u distance_state_classifier_endodac/scripts/train_appearance_consistency.py \
  --preflight-only --output-dir results/distance_experiment_b_preflight
```

运行 3 批实际 adapter 训练验证（不会评估验证集和测试集，也不代表完成正式 B）：

```bash
$PYTHON -u distance_state_classifier_endodac/scripts/train_appearance_consistency.py \
  --device cuda --smoke-steps 3 --output-dir results/distance_experiment_b_smoke_new
```

恢复同一正式实验：

```bash
$PYTHON -u distance_state_classifier_endodac/scripts/train_appearance_consistency.py \
  --device cuda \
  --resume distance_state_classifier_endodac/runs/endodac_experiment_b_appearance_consistency_20260907/last.pt
```

恢复时会检查 checkpoint 的配置与数据审计是否一致；需要保留同一输出目录中的 `best.pt`。

## 训练结束后如何比较 A/B

对同一批 122 张测试图同时运行 A、B，使用既有的固定干预脚本。不能用测试结果反复挑选 epoch 或调整增强参数。

```bash
$PYTHON scripts/diagnose_distance_shortcuts.py \
  --checkpoint distance_state_classifier_endodac/runs/endodac_encoder_multiscale_adapter_reweight_soft_392/best.pt \
  --checkpoint distance_state_classifier_endodac/runs/endodac_experiment_b_appearance_consistency_20260907/best.pt \
  --csv auto_labeling_project/data/final_dataset/test_labels.csv \
  --output-dir results/distance_experiment_b_comparison_20260907 \
  --device cuda --save-examples 12
```

汇总 A/B 指标并生成训练曲线与对比图：

```bash
$PYTHON scripts/summarize_distance_experiment_b.py
```

结果位于 `results/distance_experiment_b_comparison_20260907/experiment_b_report.md`。

一起看以下结果，而不是只追求更低的预测翻转率：

1. 原图 accuracy、macro-F1、各类别召回率，尤其是 TooFar 是否进一步漏检。
2. 7 种颜色干预下的 accuracy、macro-F1、预测翻转率及概率 TV；背景模糊单独作为较强压力测试。
3. 是否存在模型大量输出 Good，从而表面上“不再随颜色变化”的情况。
4. 预览中的器械细节、轮廓保护带和 mask 质量。

测试集 TooFar 只有 10 张：一张图就相当于召回率 10 个百分点。结果来自单个假模视频，不能据此声称真实手术泛化已解决。真实视频保持独立测试，后续重点检查手术片段中的类别召回、误判持续时间和医生撤器械时的行为。

## 验证代码

```bash
$PYTHON -m unittest discover -s tests -p 'test_appearance_consistency.py' -v
$PYTHON -m unittest discover -s tests -p 'test_distance_shortcut_diagnostics.py' -v
```

覆盖：区域外像素与轮廓不变、背景和器械分区独立、空/细 mask 行为、确定性增强、实际原图预处理一致、JS 稳定性与双向梯度、类别权重，以及训练确实更新参数而评估不更新参数。
