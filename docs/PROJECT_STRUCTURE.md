# Nasalendo Seg — YOLO11-seg 手术器械分割项目

## 文件结构

```
nasalendo_seg/
├── videos/                      # 录制的内窥镜原始视频
├── frames/                      # 从视频中均匀抽取的原始帧
├── annotations/                 # X-AnyLabeling 标注工作目录
│   ├── positive/                # 有器械的正样本（帧图片 + JSON 标注）
│   └── negative/                # 没有器械的负样本（仅帧图片）
├── dataset/                     # 最终 YOLO11-seg 格式数据集
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── scripts/                     # 数据处理脚本（抽帧、格式转换、数据集拆分等）
├── configs/                     # 训练配置文件（data.yaml 等）
├── runs/                        # 训练输出（模型权重、日志、指标等）
├── weights/                     # 预训练权重
├── docs/                        # 项目文档与说明
└── legacy/                      # 旧版代码与训练文件（归档）
```

## 冷启动工作流

| 步骤 | 目录 | 说明 |
|------|------|------|
| 1. 准备视频 | `videos/` | 将内窥镜录制视频放入此处 |
| 2. 均匀抽帧 | `frames/` | 从 `videos/` 中跨时间段均匀抽帧，输出到此处 |
| 3. 标注 | `annotations/` | 将帧复制到此处，用 X-AnyLabeling 进行多边形标注 |
| 4. 格式转换 + 拆分 | `dataset/` | JSON → YOLO seg txt，拆分为 train/val |
| 5. 训练 | `runs/` | 使用 YOLO11-seg 进行实例分割训练 |
