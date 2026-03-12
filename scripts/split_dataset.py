#!/usr/bin/env python3
"""
将正样本拆分为 train/val (8:2) 并重新编号，复制到 dataset/ 目录。

用法:
    python scripts/split_dataset.py
"""

import os
import random
import shutil
from pathlib import Path

# ===== 配置 =====
POSITIVE_DIR = "annotations/positive"
DATASET_DIR = "dataset"
TRAIN_RATIO = 0.8
SEED = 42
# ================


def main():
    src = Path(POSITIVE_DIR)
    dst = Path(DATASET_DIR)

    # 找到所有 png+txt 配对
    png_files = sorted(src.glob("*.png"))
    stems = []
    for p in png_files:
        if (src / f"{p.stem}.txt").exists():
            stems.append(p.stem)

    print("=" * 50)
    print("📊 数据集拆分 (train:val = 8:2)")
    print("=" * 50)
    print(f"📂 来源: {POSITIVE_DIR}")
    print(f"📂 目标: {DATASET_DIR}")
    print(f"📄 有效样本: {len(stems)}")

    # 随机拆分
    random.seed(SEED)
    shuffled = stems.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * TRAIN_RATIO)
    train_stems = shuffled[:n_train]
    val_stems = shuffled[n_train:]

    print(f"✂️  训练集: {len(train_stems)}, 验证集: {len(val_stems)}")

    # 清空目标目录
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = dst / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            if f.name != ".gitkeep":
                f.unlink()

    # 复制并重新编号
    def copy_split(stem_list, split_name):
        img_dst = dst / "images" / split_name
        lbl_dst = dst / "labels" / split_name
        for idx, stem in enumerate(stem_list, 1):
            new_name = f"{split_name}_{idx:04d}"
            shutil.copy2(src / f"{stem}.png", img_dst / f"{new_name}.png")
            shutil.copy2(src / f"{stem}.txt", lbl_dst / f"{new_name}.txt")

    print("\n📦 复制并重新编号...")
    copy_split(train_stems, "train")
    copy_split(val_stems, "val")

    # 统计
    print(f"\n✅ 完成！")
    print(f"   dataset/images/train/ → {len(train_stems)} 张 (train_0001.png ~ train_{len(train_stems):04d}.png)")
    print(f"   dataset/images/val/   → {len(val_stems)} 张 (val_0001.png ~ val_{len(val_stems):04d}.png)")
    print(f"   dataset/labels/train/ → {len(train_stems)} 个")
    print(f"   dataset/labels/val/   → {len(val_stems)} 个")


if __name__ == "__main__":
    main()
