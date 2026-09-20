# 假模与真实手术混合的整图距离分类

本实验移除头部检测，直接用完整光学视野RGB训练三类ConvNeXt-Tiny。没有器械掩码输入、头部裁剪、头部尺度特征。

## 训练前固定划分

- 假模沿用原始完整CSV：训练956、验证175、测试122，按视频分开。不是人工框实验94/34的小子集。
- 真实手术采用20260919人工确认的824张关键帧和帧标签。
- 真实验证病例：case_0002、case_0008、case_0010、case_0014，共166张（Far/Good/Close=40/66/60）。其余12病例658张训练。
- 在所有四病例组合中，仅根据标签计数选择各类别数量最接近20%的组合，要求训练和验证每类至少10张；并列按病例名排序。没有用模型结果选择病例。
- 训练集1614张；混合验证341张，分别报告两个来源；另有假模测试122张。
- 病例、片段、假模视频和解码图像哈希已检查，无跨划分重叠。

划分存放于`datasets/mixed_fullframe_distance_20260919/`；manifest包含源图像路径、标签、来源、哈希和所有划分。inputs是完整FOV等比例填充384的缓存，未提取头部窗口。

## 模型和训练

ImageNet预训练ConvNeXt-Tiny，先冻结主干3轮，再微调全部网络。主干LR1e-5，分类头LR3e-4，AdamW、余弦学习率、batch16、最多40轮、patience10、seed42。训练使用AMP。加权交叉熵的权重只由训练集类别计数计算。

颜色抖动、模糊、噪声增强；没有随机裁剪、缩放、翻转或旋转。最佳权重按两个来源验证Macro-F1的等权平均选择。测试集不参与模型选择。训练前参数保存在结果目录config.json。

## 运行

```bash
python scripts/train_mixed_fullframe_distance.py prepare
python scripts/train_mixed_fullframe_distance.py train
```

训练输出：`results/convnext_mixed_fullframe_20260919/`。

- `best.pt`、`config.json`：权重及设置。
- `history.json`、`training.log`：训练过程。
- `results.json`、`predictions.csv`：各来源、逐病例及逐图评估。
- `report.md`、`validation_gallery.html`：报告和真实验证图像。

独立图像推理（输入应已裁到完整光学FOV，不能输入头部ROI）：

```bash
python scripts/infer_mixed_fullframe_distance.py /absolute/path/to/fov_image.png
```

需要与训练相同的等比例填充/ImageNet归一化，不使用早期分类器的直接拉伸。真实图像的黑边裁剪沿用先前光学标定，相关坐标已记录在关键帧数据manifest中。

## 限制

真实病例留出用于开发验证并参与早停，不是独立最终临床测试。此前头部方案已在这些帧上做过诊断。相较旧方案，本轮同时改变数据、验证子集、训练方式和输入，不能将效果差异单独解释为混合训练带来的收益。没有改动实时控制程序。
