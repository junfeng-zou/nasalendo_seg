# 距离状态分类器：数据位置与干预诊断

## 训练数据在哪里

两个分类器已检查权重中的 `config.data` 均指向下列 CSV。路径相对于仓库根目录。

| 内容 | 路径 | 当前数量 |
|---|---|---:|
| 实际输入图像 | `auto_labeling_project/data/frames_cropped/<video_id>/frame_*.png` | CSV 指定使用哪些帧 |
| 训练名单及标签 | `auto_labeling_project/data/final_dataset/train_labels.csv` | 956 |
| 验证名单及标签 | `auto_labeling_project/data/final_dataset/val_labels.csv` | 175 |
| 测试名单及标签 | `auto_labeling_project/data/final_dataset/test_labels.csv` | 122 |
| 导出时的数据统计 | `auto_labeling_project/data/final_dataset/dataset_summary.json` | 总计 1253 个有效分类样本 |
| 人工复核后的标注记录（导出来源） | `auto_labeling_project/data/filtered_labels/labels_reviewed.jsonl` | 包含未用于三分类的记录 |
| 已缓存的器械分割 | `distance_state_classifier_endodac/mask_cache/yolo11s_seg_formal/` | 按图像绝对路径 SHA-1 命名 |

CSV 中 `image_path` 是实际读取的图片，`label` 是 `TooFar / Good / TooClose`；还有 `video_id`、`frame_index`、`time_sec` 等字段。图像没有额外复制成 train/val/test 三个文件夹。

当前训练视频为 `bend_data3` 和 `straight_data1` 至 `straight_data5`；验证视频为 `bend_data1`，测试视频为 `bend_data2`。三组视频无重叠，但不等于跨病例或假模到临床的独立验证。

## 已实现的干预

入口：`scripts/diagnose_distance_shortcuts.py`。仅离线推理，不训练、不修改源图或权重、不连接机器人。

对同一图像使用相同的原始 mask，生成以下配对样本：

| 变体 | 改动 | 保留 |
|---|---|---|
| `original` | 无 | 全部 |
| `bg_desaturate` | 背景 Lab 色度清零 | 器械、边界保护带、空间位置及尺寸 |
| `bg_chroma_plus/minus` | 背景 Lab 的 a/b 平面旋转 ±角度 | 同上，转换前的 L 通道 |
| `fg_desaturate` | 器械内部去色 | 器械轮廓、背景、位置及尺寸 |
| `fg_chroma_plus/minus` | 器械内部色度旋转 | 同上，转换前的 L 通道 |
| `all_desaturate` | 有效视野内部去色 | 空间位置及尺寸 |
| `bg_texture_blur` | 背景归一化高斯模糊 | 器械及其边界保护带 |

区域编辑在原图坐标系中进行，没有裁剪放大或几何变换。与原始 mask 边界默认保留 5 像素保护带。黑色边界通过最大非黑外轮廓估计 FOV 后排除，必须检查该估计是否合理。

**局限：** Lab 转回 RGB 时色域裁剪和量化可能稍微改变明度。背景模糊会损失纹理，也可能影响组织细节，属于强干预；不能默认标签必然不变。金属接近灰色时色相旋转可能几乎不起作用，因此要同时检查实际像素变化量。所有结论都取决于 mask 正确；当前诊断不会重新分割变色后的图像，也不测试分割模型本身。

这里没有通过生成模型合成图像，也没有把器械置黑或擦除。抓钳开合和出平面旋转请用真实采集的配对片段，不能靠二维仿射变换可靠模拟。

## 运行

在仓库根目录执行，使用已有 `nasalendo_seg` 环境。

```bash
conda run --no-capture-output -n nasalendo_seg python scripts/diagnose_distance_shortcuts.py \
  --max-samples 12 \
  --save-examples 4
```

默认比较当前磁盘上的两个权重：

```text
distance_state_classifier/runs/final_convnext_tiny_384_v2/best.pt
distance_state_classifier_endodac/runs/endodac_encoder_multiscale_adapter_reweight_soft_392/best.pt
```

第二个权重的内部 `model_name` 为 `endodac_encoder_classifier`，是普通多层特征版本，**不是 mask-aware**。目前 `endodac_encoder_maskaware_multiscale_soft_392` 下存在评估结果，但没有找到其权重。脚本会把实际模型类型写入报告；可以通过 `--checkpoint` 指定其他已训练权重，支持 mask-aware。

完整检查 122 帧测试集：

```bash
conda run --no-capture-output -n nasalendo_seg python scripts/diagnose_distance_shortcuts.py \
  --max-samples 0 \
  --save-examples 12
```

只测一个模型并减小变色幅度：

```bash
conda run --no-capture-output -n nasalendo_seg python scripts/diagnose_distance_shortcuts.py \
  --checkpoint distance_state_classifier/runs/final_convnext_tiny_384_v2/best.pt \
  --hue-degrees 15 \
  --max-samples 0
```

输出默认在 `results/distance_shortcut_diagnostics_<时间>/`。也可指定 `--output-dir`，但该目录必须不存在，以免覆盖旧实验。CUDA 不可用时自动使用 CPU；可用 `--threads` 和 `--batch-size` 控制资源。

当前训练/测试图片已经在 `frames_cropped` 中，**不要加 `--apply-crop`**。该参数仅适用于与训练配置一致的原始采集帧；不同来源的临床视频应先确认视野范围与缩放方式。

## 如何看结果

1. 打开 `gallery.html`：浏览器展示按最大概率变化排序的样本、mask 叠加图、原图和所有干预。先看分割是否正确，干预是否改到了器械或明显破坏了有用信息。
2. 看 `report.md`：每个模型、每种干预的总体指标。
3. 打开 `predictions.csv`：按 `variant` 筛选，再看 `flipped`、`probability_tv`、`original_confidence`；可按 `image_path` 回看原图。
4. 查看 `summary.json`：包含混淆矩阵、逐视频统计、实际权重与训练数据路径。
5. 查看 `skipped.csv`：缺失、空或尺寸不匹配的 mask 会跳过，不能算作“模型稳定”。

| 指标 | 含义 |
|---|---|
| `flip_rate` | 干预后原始预测类别与原图不同的比例 |
| `mean_probability_tv` | 概率分布总变差：`0.5 * sum(abs(p_changed - p_original))`，范围 0–1 |
| `correct_to_wrong_rate` | 原图预测正确的样本中，干预后变错的比例 |
| `accuracy_delta` | 相同有效样本上，干预准确率减去原图准确率 |
| `opposite_extreme_flip_rate` | 原图与干预预测在 TooFar 和 TooClose 两端直接互换的比例，不等于实际机器人动作率 |
| `mean_changed_pixel_fraction` | 实际有像素变化的图像面积比例 |
| `mean_pixel_delta_255` | 干预区域内的平均绝对像素变化，范围 0–255 |

例：原图概率 `[0.05, 0.90, 0.05]`，只改背景后变为 `[0.70, 0.25, 0.05]`（类别顺序 TooFar/Good/TooClose）。预测从 Good 变成 TooFar，TV 为 0.65。如果原标签是 Good，这也是一次“原本正确→错误”。

没有公认的单一翻转率阈值能证明捷径。温和外观干预在多种病例上持续引起变化，是更有力的线索；仅在强模糊或不自然颜色下变化，也可能只是分布外扰动。平滑后的状态可能掩盖敏感性，因此诊断使用未平滑概率。帧之间相关，小样本百分比不能当作临床统计结论。

## 用真实手术图像检查

准备 CSV，至少包含 `image_path`。推荐同时提供人工 `label`、手术/病例 `video_id` 和人工检查过的 `mask_path`：

```csv
image_path,mask_path,label,video_id
/path/to/frame_001.png,/path/to/mask_001.png,Good,case_01
```

```bash
conda run --no-capture-output -n nasalendo_seg python scripts/diagnose_distance_shortcuts.py \
  --csv /path/to/clinical_diagnostic.csv \
  --max-samples 0
```

显式 `mask_path` 优先于缓存，mask 应与对应图像分辨率完全一致。多器械帧需明确使用哪一个目标；默认已有缓存可能是多器械并集。无人工标签也能算预测翻转和概率变化，但不报告准确率或“原本正确→错误”。这个脚本不会自动把病例帧加入训练集。

建议对原场景和临床数据分别做同样干预，再决定是否训练背景随机化、灰度或参考图像条件下的模型。灰度推理实验本身不能替代灰度重训练对照。
