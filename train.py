#!/usr/bin/env python3
"""
YOLO11-seg 手术器械分割 - 正式训练脚本。
针对内窥镜场景做了定制化数据增强:
  - 高光/反光模拟 (specular reflection)
  - 动态模糊 (motion blur)
  - 颜色抖动，尤其红色通道 (出血模拟)

用法:
    python train.py                          # 默认: yolo11s-seg, imgsz=1024
    python train.py --imgsz 640              # 更小图像尺寸 (更快训练)
    python train.py --resume                 # 从上次中断处恢复训练
"""

import argparse
from ultralytics import YOLO
import torch


# ──────── 内窥镜定制 Albumentations 增强 ────────
def patch_albumentations():
    """
    Monkey-patch Ultralytics 内置的 Albumentations 类，
    注入内窥镜场景专用的数据增强管线。
    """
    try:
        import albumentations as A
        from ultralytics.data import augment as ultralytics_augment

        class EndoscopeAlbumentations:
            """为内窥镜手术场景定制的 Albumentations 增强管线。"""

            def __init__(self, p=1.0, **kwargs):
                self.p = p
                self.transform = A.Compose([
                    # === 1. 高光 / 镜面反射模拟 ===
                    A.RandomBrightnessContrast(
                        brightness_limit=0.4,     # 模拟局部强光
                        contrast_limit=0.3,
                        p=0.4,
                    ),
                    A.RandomToneCurve(scale=0.2, p=0.3),  # 模拟非均匀光照

                    # === 2. 动态模糊 (Motion Blur) ===
                    A.OneOf([
                        A.MotionBlur(blur_limit=(5, 15), p=1.0),      # 运动拖影
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),     # 失焦模糊
                        A.MedianBlur(blur_limit=5, p=1.0),            # 介质散射
                    ], p=0.35),

                    # === 3. 颜色抖动 - 偏红色通道 (出血模拟) ===
                    A.RGBShift(
                        r_shift_limit=30,         # 红色通道大幅波动
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=0.35,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=0.3,
                    ),

                    # === 4. 额外的鲁棒性增强 ===
                    A.CLAHE(clip_limit=4.0, p=0.2),                   # 局部对比度均衡
                    A.GaussNoise(var_limit=(10, 50), p=0.2),          # 传感器噪声
                    A.ImageCompression(quality_lower=60, p=0.15),     # JPEG 压缩伪影
                ])
                print("✅ 内窥镜定制 Albumentations 增强已启用:")
                print("   - 高光/反光模拟 (RandomBrightnessContrast + RandomToneCurve)")
                print("   - 动态模糊 (MotionBlur / GaussianBlur / MedianBlur)")
                print("   - 红色通道颜色抖动 (RGBShift + HSV)")
                print("   - CLAHE / GaussNoise / ImageCompression")

            def __call__(self, labels):
                im = labels.get("img")
                if im is None:
                    return labels
                transformed = self.transform(image=im)
                labels["img"] = transformed["image"]
                return labels

        # 替换 Ultralytics 内部的 Albumentations 类
        ultralytics_augment.Albumentations = EndoscopeAlbumentations
        print("🔧 已 patch Ultralytics Albumentations 管线\n")

    except ImportError:
        print("⚠️  albumentations 未安装，将仅使用 YOLO 内置增强。")
        print("   安装: pip install albumentations\n")


def main():
    parser = argparse.ArgumentParser(description="YOLO11-seg 训练")
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt",
                        help="预训练模型 (默认: yolo11s-seg.pt)")
    parser.add_argument("--data", type=str, default="configs/data.yaml",
                        help="数据集配置文件")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="输入图像尺寸 (默认: 1024)")
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size (默认: 16)")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备 (默认: 0)")
    parser.add_argument("--resume", action="store_true",
                        help="从上次训练中断处恢复")
    parser.add_argument("--name", type=str, default=None,
                        help="实验名称 (默认: 根据模型自动生成)")
    args = parser.parse_args()

    # 实验名称
    if args.name is None:
        model_tag = args.model.replace(".pt", "").replace("-", "_")
        args.name = f"{model_tag}_formal"

    # 注入内窥镜定制增强
    patch_albumentations()

    # 系统信息
    print("=" * 60)
    print("🚀 YOLO11-seg 正式训练 (内窥镜定制增强)")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存: {vram:.1f} GB")
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    batch_str = "自动" if args.batch == -1 else str(args.batch)
    print(f"Epochs: {args.epochs}, Batch: {batch_str}, ImgSz: {args.imgsz}")
    print("=" * 60)

    # 加载模型
    if args.resume:
        print("\n📦 恢复训练...")
        model = YOLO(f"runs/segment/{args.name}/weights/last.pt")
    else:
        print(f"\n📦 加载预训练模型: {args.model}")
        model = YOLO(args.model)

    # 训练
    print("\n🎯 开始训练...\n")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=30,
        save=True,
        device=args.device,
        workers=8,
        project="runs",
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        verbose=True,
        seed=42,
        val=True,
        plots=True,
        save_period=10,
        # ── 内置几何增强 ──
        mosaic=1.0,
        close_mosaic=15,          # 最后 15 epoch 关闭 mosaic
        mixup=0.15,               # 轻度 mixup
        degrees=15.0,             # 旋转角度 (内窥镜方向变化大)
        scale=0.5,                # 缩放范围
        fliplr=0.5,               # 水平翻转
        flipud=0.0,               # 不做上下翻转 (内窥镜有方向性)
        # ── 内置颜色增强 (加强版) ──
        hsv_h=0.02,               # 色调波动
        hsv_s=0.6,                # 饱和度波动 (内窥镜色彩变化剧烈)
        hsv_v=0.4,                # 明度波动 (光照变化剧烈)
    )

    # 获取实际保存路径
    save_dir = model.trainer.save_dir if hasattr(model, 'trainer') else f"runs/segment/{args.name}"

    # 验证
    print("\n📊 验证最佳模型...")
    best_pt = f"{save_dir}/weights/best.pt"
    best_model = YOLO(best_pt)
    metrics = best_model.val(data=args.data)

    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print("=" * 60)
    print(f"Box  mAP50:    {metrics.box.map50:.4f}")
    print(f"Box  mAP50-95: {metrics.box.map:.4f}")
    print(f"Mask mAP50:    {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    print(f"\n模型保存在: {save_dir}/weights/")


if __name__ == "__main__":
    main()

