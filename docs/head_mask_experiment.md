# 器械掩码引导的头部检测和 B/C 分类实验

## 目的与实现

验证预测的器械掩码能否帮助定位头部、提高自动框下的深度状态分类。第一轮采用固定强度的软背景抑制：

`input = round(original_RGB × (0.5 + 0.5 × predicted_mask))`

器械区域保留原像素，背景保留一半亮度。该方式将掩码作为输入预处理先验，不增加网络通道或分支；不是RGB+mask四通道网络。这样可以沿用原模型结构和初始化，不把更多模型参数造成的变化混入结果。

掩码完全来自实时程序默认的YOLO11l分割权重，分割器冻结。conf=0.8、IoU=0.7、imgsz=1024；retina_masks=True使掩码直接处于原图坐标，合并所有预测器械实例。未使用头部框选择分割实例、未人工修补、未做时序跟踪。空掩码仍生成半亮度原图。

## 比较设置

沿用当前manifest的annotation_split。头部检测训练/验证144/36张；分类训练/验证94/34张有头部帧。空头部标注作为检测负样本，分类不分配距离真值。

检测器保持原YOLO11s结构、COCO初始化、100轮上限、patience25及其他全部训练参数，仅改变输入图像。分类保持冻结ConvNeXt和FusionHead，与原B/C采用相同随机初始化、seed42/43/44、优化器、早停和尺度归一化。没有训练原分割模型。

|检测框来源|分类输入|说明|
|---|---|---|
|人工框|RGB/掩码引导|隔离分类输入影响|
|原RGB检测器|RGB|原自动框基线|
|原RGB检测器|掩码引导|只改变分类输入与分类训练|
|新掩码引导检测器|RGB|只改变检测模块|
|新掩码引导检测器|掩码引导|完整组合|

每组均测试B和C，seed42为主结果，另报告三个分类种子均值。B没有显式尺度输入，仍有裁剪像素；C保留框宽高/FOV尺度。

所有自动框均conf=0.25、NMS IoU=0.7、最高置信度，不使用人工框做候选筛选。分类主分母34张，检测不到记Invalid并计入FN。检测成功子集另列；同时列出两种检测器共同检出的相同子集，避免覆盖率差异造成误读。

## 运行与产物

使用nasalendo_seg环境，在项目根目录运行：

```bash
python scripts/run_head_mask_experiment.py prepare
python scripts/run_head_mask_experiment.py detector
python scripts/run_head_mask_experiment.py classifier
python scripts/evaluate_head_mask_experiment.py
```

准备完成后，检测训练与分类训练可以独立运行；评估必须等待两者完成。完成目录会拒绝覆盖。

产物位于 `results/head_mask_experiment_20260918/`：

- `prepared.json`：分割来源、参数、各图像掩码和原始文件哈希。
- `mask_preview.html`：原图、预测mask、实际输入的并排预览。
- `detector_train/weights/best.pt`：新检测器。
- `classifier/runs/{B_roi,C_roi_scale}/seed_{42,43,44}/best.pt`：掩码引导分类头。
- backbone沿用 `results/convnext_manual_head_abc_20260917/frozen_backbone.pt`。
- `report.md`、`comparison_gallery.html`、`comparison_summary.json`：因子对照结果。

## 解释限制

只检验50%背景保留这一种mask使用方式，不把结论推广到四通道融合、特征级注意力或所有掩码方法。同视频混合验证且验证集参与模型选择，不能证明真实手术泛化；分割器训练数据与本批图像的关系未构成独立链路测试。原标签、旧权重与实时程序不修改。
