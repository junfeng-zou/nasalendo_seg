# 内窥镜器械距离状态单帧分类器

模型结构、训练策略和推理输出字段的详细说明见 [MODEL.md](MODEL.md)。

本模块实现单帧内窥镜器械距离状态分类：

```text
训练: TooFar / Good / TooClose 三分类
推理: 规则 + 低置信度阈值输出 Invalid
控制: 使用短窗口时间平滑，避免单帧抖动
```

当前实现支持两类模型：

```text
1. 自包含模型：resnet18_small / tiny_cnn
2. timm 预训练模型：timm:convnext_tiny / timm:tf_efficientnetv2_s / timm:resnet18
```

默认配置使用：

```yaml
model:
  name: timm:convnext_tiny
  pretrained: true

train:
  head_lr: 0.0003
  backbone_lr: 0.00001
  freeze_backbone_epochs: 5
  grad_clip_norm: 1.0
```

预训练模型微调时不要让 backbone 和分类头使用同样大的学习率。分类头可以学得快一些，backbone 应该小学习率微调，否则容易在解冻后破坏预训练特征。

默认 `num_workers: 0` 使用单进程加载数据；可根据 CPU 和存储吞吐量调整为 `num_workers: 4` 或通过命令行覆盖。

## 目录结构

```text
distance_state_classifier/
├── configs/
│   └── distance_state_config.yaml
├── scripts/
│   ├── train.py
│   ├── infer_image.py
│   ├── infer_video.py
│   └── infer_camera.py
└── src/
    ├── config.py
    ├── dataset.py
    ├── metrics.py
    ├── model.py
    ├── predictor.py
    └── transforms.py
```

## 训练

```bash
cd nasalendo_seg
python distance_state_classifier/scripts/train.py \
  --config distance_state_classifier/configs/distance_state_config.yaml
```

常用覆盖参数：

```bash
python distance_state_classifier/scripts/train.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --device cuda \
  --batch-size 32 \
  --num-workers 4
```

如果想快速对比其他预训练 backbone：

```bash
python distance_state_classifier/scripts/train.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --model timm:tf_efficientnetv2_s \
  --output-dir distance_state_classifier/runs/efficientnetv2_s_384 \
  --device cuda \
  --batch-size 16 \
  --num-workers 4
```

如果只是测试代码流程，不想下载预训练权重：

```bash
python distance_state_classifier/scripts/train.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --model timm:resnet18 \
  --no-pretrained \
  --epochs 1 \
  --input-size 128 \
  --device cuda
```

## Leave-One-Video-Out 交叉验证

生成按视频留一法 splits：

```bash
python distance_state_classifier/scripts/make_video_splits.py \
  --config distance_state_classifier/configs/distance_state_config.yaml
```

输出目录：

```text
distance_state_classifier/splits/leave_one_video_out/
```

训练所有 fold：

```bash
python distance_state_classifier/scripts/train_crossval.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --device cuda \
  --batch-size 32 \
  --num-workers 4
```

只训练某一个测试视频 fold，例如 `bend_data2`：

```bash
python distance_state_classifier/scripts/train_crossval.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --only-fold bend_data2 \
  --device cuda \
  --batch-size 32 \
  --num-workers 4
```

每个 fold 的结果会保存在：

```text
distance_state_classifier/runs/crossval_convnext_tiny_384/test_<video_id>/
```

汇总结果：

```text
distance_state_classifier/runs/crossval_convnext_tiny_384/crossval_summary.json
```

训练输出默认保存到：

```text
distance_state_classifier/runs/resnet18_small_384/
```

主要文件：

```text
best.pt
last.pt
metrics.json
confusion_matrix.csv
class_mapping.json
```

## 单张图片推理

```bash
python distance_state_classifier/scripts/infer_image.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --checkpoint distance_state_classifier/runs/resnet18_small_384/best.pt \
  --image auto_labeling_project/data/frames_cropped/bend_data1/frame_000000.png
```

如果输入是原始未裁剪图，可加：

```bash
--apply-crop
```

## 视频推理

```bash
python distance_state_classifier/scripts/infer_video.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --checkpoint distance_state_classifier/runs/resnet18_small_384/best.pt \
  --video auto_labeling_project/data/raw_videos/bend_data1.avi \
  --output distance_state_classifier/runs/bend_data1_predictions.jsonl \
  --apply-crop
```

视频推理会输出每帧的：

```text
raw_label
raw_confidence
state
smoothed_state
probabilities
```

## 实时摄像头推理

使用最终训练好的权重读取摄像头实时推理：

```bash
python distance_state_classifier/scripts/infer_camera.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --checkpoint distance_state_classifier/runs/final_convnext_tiny_384_v2/best.pt \
  --camera 0 \
  --device cuda \
  --apply-crop
```

如果摄像头已经输出裁剪后的有效视野，不要加 `--apply-crop`。

常用参数：

```text
--width 1920 --height 1080 --fps 30   请求摄像头分辨率和帧率
--display-crop                        只显示裁剪后的有效视野
--infer-every 2                       每 2 帧推理一次，用于降低延迟
--save-video path.avi                 保存带预测文字的视频
--save-jsonl path.jsonl               保存逐帧推理结果
--no-display                          不打开预览窗口
```

退出预览窗口：

```text
q 或 Esc
```

## Invalid 策略

第一版模型只训练三分类：

```text
TooFar / Good / TooClose
```

推理时满足以下情况会输出 `Invalid`：

```text
1. 最大 softmax 置信度低于 confidence_threshold
2. 可选: 后续接入 YOLO 器械检测，未检测到器械时输出 Invalid
```

默认阈值在配置文件中：

```yaml
inference:
  confidence_threshold: 0.55
```

## 类别映射

默认类别顺序：

```text
0: TooFar
1: Good
2: TooClose
```

训练和推理必须使用同一个 `class_mapping.json`。
