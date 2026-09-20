# 自动头部检测＋混合数据分类实验

本轮固定 `results/forceps_head_yolo11s_20260917/train/weights/best.pt`，用自动框重新训练 B/C。真实手术数据只有逐帧距离标签，没有人工头部框，因此没有用距离标签重训检测器。

## 固定协议

- 源划分：`datasets/mixed_fullframe_distance_20260919/manifest.json`；真实验证病例为 case_0002、0008、0010、0014。
- 自动框：YOLO11s，1024，FP32，conf=0.25，NMS IoU=0.7，max_det=100，取最高置信度框。真实帧复用相同协议缓存并检查图像哈希；假模重新推理。
- B：头部框扩大1.2倍，等比例填充384，RGB输入。
- C：同样RGB输入，附加未扩大框的宽/光学视野直径、高/光学视野直径；仅用训练有框样本拟合均值/标准差。
- B/C共享相同容量的 FusionHead128；B的两个尺度输入固定为0。ImageNet初始化主干，冻结3轮后微调，最多40轮，10轮早停。类别权重来自有效训练样本。
- 无框的训练帧不参加头部分类训练；无框验证帧为 Invalid，保留在完整评估分母内。另报有框子集和固定整图模型回退方案。
- 不使用器械分割，不改实时程序。

## 运行

```bash
PYTHON=python
$PYTHON scripts/train_mixed_head_distance.py prepare
$PYTHON scripts/train_mixed_head_distance.py B
$PYTHON scripts/train_mixed_head_distance.py C
$PYTHON scripts/train_mixed_head_distance.py evaluate
$PYTHON scripts/check_mixed_head_result.py
```

已有缓存和权重时，prepare/train 会拒绝覆盖。evaluate 会更新本轮结果报告。

单图输入须已裁成与训练相同的光学视野；C 的直径使用该输入坐标下的光学视野标定结果：

```bash
$PYTHON scripts/infer_mixed_head_distance.py IMAGE.png --arm C --fov-diameter DIAMETER_IN_PIXELS --fallback-fullframe
```

这只是单图验证入口，每次执行会加载模型，不代表稳态实时延迟。实时集成需持久加载模型；本轮未修改实时程序。

## 解释边界

检测器训练图与假模验证集存在24张重合，假模验证不代表完全未见的级联评估。真实病例仍分离，但验证集用于选权重，不是最终独立测试集。B/C只使用724张有框训练图（585假模＋139真实），整图使用1614张；分类头也不同，所以比较的是当前可用方案，不能视作控制所有变量的纯裁剪消融。框覆盖率并非有真值支持的检测召回率。单种子结果需要更多独立病例确认。

结果：`results/convnext_mixed_head_20260919/report.md`、`comparison_gallery.html`、`results.json`、`predictions.json`，权重在 `B/best.pt` 和 `C/best.pt`。
