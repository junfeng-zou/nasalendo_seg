# YOLO11s 头部检测与固定 C 的替换对照

本实验训练单类别 `forceps_head` 检测器，再将原 C 的人工框替换为自动框。C 的 ConvNeXt backbone、分类头和尺度归一化全部冻结。

## 数据和协议

- 标注来源：`annotation_projects/forceps_head_pilot_20260907/manifest.csv` 的 `annotation_split`，不用历史 `original_split`。
- 训练144张：94正、50空标注；验证36张：34正、2空标注。
- 矩形兼容 X-AnyLabeling 两点/四顶点格式，转换为 YOLO 单类框；空 JSON 生成空 txt。原图和 JSON 不修改。
- YOLO11s COCO 初始化，1024 输入，batch8，最多100轮、patience25，seed42。检测验证指标选 best，不根据后续分类结果挑权重。
- 推理固定 conf=0.25、NMS IoU=0.7，选置信度最高框。不能依据人工框 IoU 挑候选，也不能在漏检时回填人工框。
- C：`results/convnext_manual_head_abc_20260917/runs/C_roi_scale/seed_42/best.pt`，并加载同目录实验根的 `frozen_backbone.pt`。
- 两种框均扩1.2倍后裁剪、等比例填充384；尺度由未扩大框宽高除以固定 FOV 直径得到。归一化只使用旧 C checkpoint 参数。
- 分类主结果：34张全部有头部验证帧，漏检输出Invalid并作为对应真值类FN。同时报告检测可用子集上的配对结果；无头部2帧仅报告检测误检。
- 主对照 FP32；FP16耗时另测，不把该耗时混称为主对照的实测延迟。

## 运行

在项目根目录、可使用 GPU 的 `nasalendo_seg` 环境：

```bash
python scripts/train_forceps_head_detector.py --prepare-only
python scripts/train_forceps_head_detector.py
python scripts/evaluate_forceps_head_replacement.py
```

初始化权重先从官方 YOLO11s 权重复制到实验根的 `initial_yolo11s.pt`。导出脚本使用 `weights/yolo11s.pt`；请先提供官方初始化权重，或在实验根目录预置 `initial_yolo11s.pt`。训练脚本遇到已存在的 `train/` 会停止，避免覆盖或混淆已有实验。

输出根目录：`results/forceps_head_yolo11s_20260917/`。

- `train/weights/best.pt`：头部检测器。
- `data_audit.json`：划分、标注、图像哈希和固定 C 权重哈希。
- `train_config.json`、`train/args.yaml`：请求参数及 Ultralytics 实际参数。
- `comparison_summary.json`、`comparison_predictions.csv/json`：指标和逐帧结果。
- `report.md`、`comparison_gallery.html`：文字报告与人工/自动框及裁剪对照。

## 边界

这是同视频混合划分的开发验证，两个模型均使用了该验证集做模型选择。不能作为独立测试或真实手术泛化证据。没有修改实时控制程序。无检测时的Invalid不能解释成Good，更不能据此生成轴向跟随指令。
