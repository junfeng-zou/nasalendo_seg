# 混合整图 RGB＋预测器械掩码实验

固定之前的 YOLO11l 分割模型，重新从相同 ImageNet 权重训练 ConvNeXt-Tiny 四通道分类器。原纯 RGB 权重保留，不使用头部检测器。

## 输入和训练

- RGB：完整光学视野，等比例填充384，ImageNet归一化。
- mask：原分辨率预测器械实例并集，最近邻缩放到与RGB相同的填充几何，第四通道为0/1。分割参数1024、conf0.8、IoU0.7、retina_masks=True、FP32，与此前掩码实验一致；本轮没有针对验证结果调整阈值。
- 首层卷积前三通道及分类头初始值与RGB基线一致；新增通道零初始化。创建新卷积时保留随机数状态，避免无意改变后续训练随机序列。
- 前3轮只训练分类头；随后主干和新增通道一起微调。种子42、40轮上限、早停10轮、学习率、类别权重与基线一致。
- 沿用基础颜色、模糊、噪声增强，仅作用于RGB；本轮没有几何变换。今后若增加几何增强，必须同步变换掩码。
- 空掩码用全零通道，不丢弃样本，不回退另一个模型。

## 划分和结果

沿用 `datasets/mixed_fullframe_distance_20260919/manifest.json`：956假模＋658真实训练，175假模＋166真实验证，122假模测试。真实验证病例为0002、0008、0010、0014；无病例/片段跨训练验证。验证用于选权重，不是临床独立测试集。

掩码缓存 `datasets/mixed_rgbmask_distance_20260920/`。

结果和权重 `results/convnext_mixed_rgbmask_20260920/`，主要阅读 `report.md`、`comparison_gallery.html`。空掩码分组比较见 `mask_diagnostics.json`，可复核完整逐帧概率 `predictions.csv`。

## 命令

```bash
PYTHON=python
$PYTHON scripts/train_mixed_rgbmask_distance.py prepare
$PYTHON scripts/train_mixed_rgbmask_distance.py train
$PYTHON scripts/check_mixed_rgbmask_result.py
$PYTHON scripts/infer_mixed_rgbmask_distance.py IMAGE_ALREADY_CROPPED_TO_OPTICAL_FOV.png
```

prepare/train拒绝覆盖已有缓存/权重。单图入口每次加载分割器和分类器，便于复核，不代表稳态实时速度。没有替换现有实时程序。若实时复用分割输出，需要保证视野裁剪、掩码坐标、阈值和分辨率一致。
