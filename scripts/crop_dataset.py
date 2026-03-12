#!/usr/bin/env python3
"""
将 dataset/ 中的图片从 1920x1080 中心裁剪为 1080x1080,
同时调整 YOLO 标注的归一化坐标。

裁剪逻辑:
  原图 1920x1080 → 左右各裁掉 420px → 1080x1080
  x_new = (x_old * 1920 - 420) / 1080
  y_new = y_old (不变)

用法:
    python scripts/crop_dataset.py
"""

import cv2
import os
from pathlib import Path

# ===== 配置 =====
DATASET_DIR = "dataset"
ORIG_W = 1920
ORIG_H = 1080
CROP_SIZE = 1080  # 目标正方形尺寸
# ================

CROP_X = (ORIG_W - CROP_SIZE) // 2  # 420


def crop_image(img_path: str) -> bool:
    """中心裁剪图片为正方形。"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ⚠️  无法读取: {img_path}")
        return False

    h, w = img.shape[:2]
    if w != ORIG_W or h != ORIG_H:
        print(f"  ⚠️  尺寸不匹配: {img_path} ({w}x{h})")
        return False

    # 中心裁剪
    cropped = img[:, CROP_X:CROP_X + CROP_SIZE]
    cv2.imwrite(img_path, cropped)
    return True


def adjust_label(txt_path: str) -> bool:
    """调整标注坐标适配裁剪后的图片。"""
    with open(txt_path, 'r') as f:
        content = f.read().strip()

    if not content:
        # 空文件 (负样本), 不需要调整
        return True

    new_lines = []
    for line in content.split('\n'):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = parts[0]
        coords = [float(x) for x in parts[1:]]

        # 每两个值一组 (x, y)
        new_coords = []
        for i in range(0, len(coords), 2):
            nx_old = coords[i]
            ny_old = coords[i + 1]

            # 转换 x 坐标: (nx * 1920 - 420) / 1080
            nx_new = (nx_old * ORIG_W - CROP_X) / CROP_SIZE
            ny_new = ny_old  # y 不变

            # 裁剪到 [0, 1]
            nx_new = max(0.0, min(1.0, nx_new))
            ny_new = max(0.0, min(1.0, ny_new))

            new_coords.extend([f"{nx_new:.6f}", f"{ny_new:.6f}"])

        new_lines.append(f"{class_id} " + " ".join(new_coords))

    with open(txt_path, 'w') as f:
        f.write("\n".join(new_lines))
        if new_lines:
            f.write("\n")

    return True


def main():
    print("=" * 50)
    print("✂️  中心裁剪 1920x1080 → 1080x1080")
    print(f"   裁剪区域: x=[{CROP_X}, {CROP_X + CROP_SIZE}]")
    print("=" * 50)

    for split in ["train", "val"]:
        img_dir = Path(DATASET_DIR) / "images" / split
        lbl_dir = Path(DATASET_DIR) / "labels" / split

        images = sorted(img_dir.glob("*.png"))
        print(f"\n📂 {split}: {len(images)} 张")

        img_ok = 0
        lbl_ok = 0
        for img_path in images:
            stem = img_path.stem
            txt_path = lbl_dir / f"{stem}.txt"

            if crop_image(str(img_path)):
                img_ok += 1
            if txt_path.exists() and adjust_label(str(txt_path)):
                lbl_ok += 1

        print(f"   ✅ 图片裁剪: {img_ok}/{len(images)}")
        print(f"   ✅ 标注调整: {lbl_ok}/{len(images)}")

    # 验证
    sample = list((Path(DATASET_DIR) / "images" / "train").glob("*.png"))[0]
    img = cv2.imread(str(sample))
    h, w = img.shape[:2]
    print(f"\n📐 验证: {sample.name} → {w}x{h}")
    print("✅ 完成!")


if __name__ == "__main__":
    main()
