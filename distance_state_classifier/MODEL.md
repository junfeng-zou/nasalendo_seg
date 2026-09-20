# distance_state_classifier 模型说明

本文档说明 `distance_state_classifier` 中训练和推理实际使用的模型、输入输出格式、训练策略以及推理后的状态判定逻辑。对应代码入口主要是：

```text
distance_state_classifier/src/model.py
distance_state_classifier/src/dataset.py
distance_state_classifier/src/transforms.py
distance_state_classifier/src/predictor.py
distance_state_classifier/scripts/train.py
distance_state_classifier/scripts/infer_image.py
distance_state_classifier/scripts/infer_video.py
distance_state_classifier/scripts/infer_camera.py
```

## 任务定义

模型做的是内窥镜单帧图像三分类，用来判断器械与目标区域的距离状态。

默认类别顺序为：

```text
0: TooFar
1: Good
2: TooClose
```

模型本身只训练这三个类别。`Invalid` 不是训练类别，而是在推理阶段根据置信度阈值和时间平滑规则派生出来的状态。

## 默认模型

默认配置位于 `configs/distance_state_config.yaml`：

```yaml
model:
  name: timm:convnext_tiny
  pretrained: true
  dropout: 0.2

image:
  input_size: 384
```

因此默认训练模型是 `timm` 提供的 `convnext_tiny`：

```text
输入:  3 x 384 x 384 RGB 图像张量
输出:  3 维 logits，对应 TooFar / Good / TooClose
权重:  使用 timm 预训练权重初始化 backbone
分类头: 替换为 3 分类输出层
dropout: 0.2
```

模型创建逻辑在 `src/model.py` 的 `build_model()` 中：

```python
timm.create_model(
    model_name,
    pretrained=bool(pretrained),
    num_classes=num_classes,
    drop_rate=float(dropout),
)
```

训练时 `pretrained: true` 表示加载预训练权重；推理时会重新构建同名网络并加载 checkpoint 中保存的 `model_state`，此时不再需要重新下载预训练权重。

## 可选模型

当前代码支持三类模型名称。

| 模型名 | 来源 | 说明 |
| --- | --- | --- |
| `timm:convnext_tiny` | timm | 默认模型，适合作为主要训练模型 |
| `timm:tf_efficientnetv2_s` | timm | 可选预训练模型，通常精度潜力较好但显存和计算量更高 |
| `timm:resnet18` | timm | 可选轻量预训练模型，适合快速实验 |
| `resnet18_small` / `small_resnet` / `resnet` | 本项目自定义 | 小型残差网络，不依赖 timm 预训练 |
| `tiny_cnn` / `tiny` | 本项目自定义 | 更轻量的 CNN，适合流程验证或低算力场景 |

### 自定义 SmallResNet

`SmallResNet` 是项目内置的小型 ResNet 风格模型：

```text
Stem:
  7x7 Conv, stride=2
  BatchNorm
  ReLU
  MaxPool

Residual stages:
  layer1: 32 通道，2 个 BasicBlock
  layer2: 64 通道，2 个 BasicBlock，stride=2
  layer3: 128 通道，2 个 BasicBlock，stride=2
  layer4: 256 通道，2 个 BasicBlock，stride=2

Head:
  AdaptiveAvgPool2d(1)
  Flatten
  Dropout
  Linear(256, num_classes)
```

每个 `BasicBlock` 包含两个 `3x3 Conv + BatchNorm`，并在通道数或步幅变化时使用 `1x1 Conv` 做残差下采样。

### 自定义 TinyCNN

`TinyCNN` 是更简单的卷积分类器：

```text
通道变化: 3 -> 32 -> 64 -> 128 -> 192
每个阶段:
  3x3 Conv, stride=2
  BatchNorm
  ReLU
  3x3 Conv, stride=1
  BatchNorm
  ReLU

Head:
  AdaptiveAvgPool2d(1)
  Flatten
  Dropout
  Linear(192, num_classes)
```

它主要适合快速跑通训练与推理流程，作为轻量 baseline。

## 输入预处理

训练和推理共用 `src/transforms.py` 中的预处理逻辑。

### 图像读取与颜色空间

图像通过 OpenCV 读取，原始格式是 BGR，随后转换为 RGB：

```text
OpenCV BGR -> RGB
```

### 裁剪

配置中预留了原始 1920x1080 图像的有效视野裁剪框：

```yaml
image:
  crop:
    enabled: false
    x_left: 362
    x_right: 1605
    y_top: 0
    y_bottom: 1080
```

训练阶段只有当 `image.crop.enabled: true` 时才会裁剪。默认 `false`，表示训练 CSV 中的图像通常已经是裁剪后的有效视野。

推理阶段由命令行参数控制：

```text
--apply-crop
```

如果输入是原始 1920x1080 帧，需要加 `--apply-crop`；如果输入已经是裁剪后的图像，不要加。

### Resize 与归一化

模型输入统一 resize 到正方形：

```text
384 x 384
```

随后做 ImageNet 归一化：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

最终张量格式为：

```text
float32, shape = [3, 384, 384], channel-first
```

### 训练增强

训练集在 `augmentation.enabled: true` 时使用轻量增强：

```yaml
brightness: 0.15
contrast: 0.15
saturation: 0.10
hue: 0.03
blur_prob: 0.15
noise_prob: 0.15
rotate_deg: 5
translate_frac: 0.04
scale_frac: 0.08
horizontal_flip_prob: 0.0
```

增强只用于训练集，验证集、测试集和推理阶段不使用随机增强。

## 训练流程

训练入口：

```bash
python distance_state_classifier/scripts/train.py \
  --config distance_state_classifier/configs/distance_state_config.yaml
```

训练数据来自配置文件中的 CSV：

```yaml
data:
  train_csv: auto_labeling_project/data/final_dataset/train_labels.csv
  val_csv: auto_labeling_project/data/final_dataset/val_labels.csv
  test_csv: auto_labeling_project/data/final_dataset/test_labels.csv
  image_column: image_path
  label_column: label
```

Dataset 会过滤掉不属于 `TooFar / Good / TooClose` 的行。

### 损失函数

训练使用交叉熵损失：

```text
CrossEntropyLoss
```

默认启用类别权重：

```yaml
train:
  use_class_weights: true
```

类别权重根据训练集每类样本数自动计算，用于缓解类别不均衡。

### 优化器与学习率

优化器是 `AdamW`：

```yaml
train:
  head_lr: 0.0003
  backbone_lr: 0.00001
  weight_decay: 0.0001
```

当代码能识别模型分类头时，会把参数分成两组：

```text
backbone: lr = 1e-5
head:     lr = 3e-4
```

这样做的目的，是让新初始化或新替换的分类头学习得更快，同时用较小学习率微调预训练 backbone，降低破坏预训练视觉特征的风险。

学习率调度器为：

```text
CosineAnnealingLR
```

调度周期 `T_max` 等于配置中的总 epoch 数。

### 冻结策略

默认前 5 个 epoch 冻结 backbone，只训练分类头：

```yaml
train:
  freeze_backbone_epochs: 5
```

从第 6 个 epoch 开始解冻 backbone，进行全模型微调。

### 其他训练设置

默认训练设置：

```yaml
train:
  seed: 42
  epochs: 60
  batch_size: 16
  num_workers: 0
  amp: true
  patience: 15
  save_every: 10
  grad_clip_norm: 1.0
```

含义如下：

| 参数 | 作用 |
| --- | --- |
| `amp` | CUDA 上启用自动混合精度 |
| `grad_clip_norm` | 梯度裁剪，限制梯度范数 |
| `patience` | 验证集 macro-F1 连续若干 epoch 无提升则 early stop |
| `save_every` | 每隔若干 epoch 保存一次中间 checkpoint |
| `seed` | 固定 Python、NumPy、PyTorch 随机种子 |

### 评价指标

训练过程中主要关注验证集：

```text
macro_f1
```

每个 epoch 会计算：

```text
accuracy
macro_f1
balanced_accuracy
loss
confusion_matrix
```

最佳模型选择规则：

```text
验证集 macro_f1 越高越好
```

训练结束后，会加载 `best.pt` 在测试集上评估。

## Checkpoint 内容

训练输出目录默认为：

```text
distance_state_classifier/runs/convnext_tiny_384/
```

主要输出文件：

```text
best.pt
last.pt
epoch_XXX.pt
metrics.json
val_confusion_matrix.csv
test_confusion_matrix.csv
test_metrics.json
class_mapping.json
```

其中 checkpoint 内部保存：

```text
model_state
optimizer_state
scheduler_state
epoch
best_metric
classes
input_size
model_name
config
```

推理最关键的是：

```text
model_state
classes
input_size
model_name
```

所以推理时不需要手动指定模型结构，只要 checkpoint 是由本项目训练脚本保存的即可。

## 推理流程

推理封装在 `src/predictor.py` 的 `DistanceStatePredictor` 中。

初始化时会执行：

```text
1. 读取 checkpoint
2. 从 checkpoint/config 获取 model_name、classes、input_size
3. 使用 build_model() 重建网络
4. 加载 model_state
5. 切换到 eval 模式
```

单帧推理流程：

```text
1. BGR 图像转 RGB
2. 可选 apply_raw_crop
3. resize 到 input_size x input_size
4. ImageNet mean/std 归一化
5. 前向计算 logits
6. softmax 得到三类概率
7. argmax 得到 raw_label 和 raw_confidence
8. 根据 confidence_threshold 生成 state
9. 使用 StateSmoother 生成 smoothed_state
```

输出字段：

| 字段 | 含义 |
| --- | --- |
| `raw_label` | softmax 概率最大的三分类标签 |
| `raw_confidence` | `raw_label` 对应的概率 |
| `state` | 置信度阈值过滤后的状态，可能是 `Invalid` |
| `smoothed_state` | 短时间窗口平滑后的状态 |
| `probabilities` | 三个训练类别的 softmax 概率 |

## Invalid 与时间平滑

默认推理配置：

```yaml
inference:
  confidence_threshold: 0.55
  smoothing_window: 5
  min_state_count: 3
  prefer_tooclose: true
```

### 置信度阈值

如果最大 softmax 概率低于阈值：

```text
raw_confidence < 0.55
```

则：

```text
state = Invalid
```

否则：

```text
state = raw_label
```

### 时间平滑

`StateSmoother` 会维护最近 `smoothing_window` 个状态，默认窗口长度为 5。

如果某个状态在窗口中出现次数达到 `min_state_count`，默认 3 次，则更新 `smoothed_state`。

当 `prefer_tooclose: true` 时，只要窗口中 `TooClose` 达到 3 次，会优先输出 `TooClose`。这是偏保守的策略，用于减少器械过近状态被短时抖动掩盖的风险。

## 推理脚本

### 单张图片

```bash
python distance_state_classifier/scripts/infer_image.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --checkpoint distance_state_classifier/runs/convnext_tiny_384/best.pt \
  --image auto_labeling_project/data/frames_cropped/bend_data1/frame_000000.png
```

如果输入是原始未裁剪图：

```bash
python distance_state_classifier/scripts/infer_image.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --checkpoint distance_state_classifier/runs/convnext_tiny_384/best.pt \
  --image path/to/raw_frame.png \
  --apply-crop
```

### 视频文件

```bash
python distance_state_classifier/scripts/infer_video.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --checkpoint distance_state_classifier/runs/convnext_tiny_384/best.pt \
  --video auto_labeling_project/data/raw_videos/bend_data1.avi \
  --output distance_state_classifier/runs/bend_data1_predictions.jsonl \
  --apply-crop
```

视频脚本会把每次推理结果按 JSONL 写出，每行对应一帧。

常用参数：

```text
--stride N        每 N 帧推理一次
--max-frames N    最多推理 N 帧
```

### 实时摄像头

```bash
python distance_state_classifier/scripts/infer_camera.py \
  --config distance_state_classifier/configs/distance_state_config.yaml \
  --checkpoint distance_state_classifier/runs/convnext_tiny_384/best.pt \
  --camera 0 \
  --device cuda \
  --apply-crop
```

常用参数：

```text
--width 1920 --height 1080 --fps 30
--display-crop
--infer-every 2
--save-video path.avi
--save-jsonl path.jsonl
--no-display
```

## 训练与推理的一致性要求

为了避免训练和部署结果不一致，需要重点保持以下内容一致：

```text
1. input_size
2. 类别顺序 classes
3. 是否对原始图像使用 crop
4. RGB/BGR 转换流程
5. ImageNet mean/std 归一化
6. checkpoint 中的 model_name 与 model_state
```

其中 `input_size`、`classes`、`model_name` 会保存在 checkpoint 中，推理代码会优先从 checkpoint 读取。裁剪策略需要根据输入图像来源手动选择是否添加 `--apply-crop`。
