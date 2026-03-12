"""
自动标注脚本：使用训练好的 YOLO11-seg 模型对正式数据帧进行推理，
将结果保存为 X-AnyLabeling 兼容的 JSON 格式。

所有帧均统一保存至: annotations/positive/formal/<video>/

用法:
    conda run -n nasalendo_seg python scripts/auto_annotate.py
"""

import json
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ──────── 配置 ────────
MODEL_PATH = "runs/segment/runs/yolo11l_seg_instrument/weights/best.pt"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRAME_DIRS = {
    "output3": PROJECT_ROOT / "frames" / "formal" / "output3",
    "output4": PROJECT_ROOT / "frames" / "formal" / "output4",
    "output5": PROJECT_ROOT / "frames" / "formal" / "output5",
    "output6": PROJECT_ROOT / "frames" / "formal" / "output6",
}

ANNO_POS_BASE = PROJECT_ROOT / "annotations" / "positive" / "formal"

CONF_THRESHOLD = 0.25  # 置信度阈值
APPROX_EPSILON_FACTOR = 0.003 # 轮廓简化的 epsilon 系数，越大点越少。0.001~0.005 是比较合理的范围


def make_xanylabeling_json(img_name, img_w, img_h, shapes):
    """生成 X-AnyLabeling 格式的 JSON 字典"""
    return {
        "version": "3.3.5",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def mask_to_shapes(result):
    """从 YOLO 推理结果中提取多边形 shapes 列表，并对点数进行简化"""
    shapes = []
    if result.masks is None:
        return shapes

    for mask_xy, cls_id, conf in zip(
        result.masks.xy, result.boxes.cls, result.boxes.conf
    ):
        if conf < CONF_THRESHOLD:
            continue
        
        # mask_xy 是 numpy array，形状 (N, 2)
        # 用 cv2.approxPolyDP 减少多边形点数
        contour = np.array(mask_xy, dtype=np.float32)
        epsilon = APPROX_EPSILON_FACTOR * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # 将 numpy array 转换回 list
        points = [[float(point[0][0]), float(point[0][1])] for point in approx_contour]
        
        if len(points) < 3:  # 多边形至少需要 3 个点
            continue
            
        label = result.names[int(cls_id)]
        shapes.append({
            "label": label,
            "score": round(float(conf), 4),
            "points": points,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "polygon",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        })
    return shapes


def main():
    model = YOLO(str(PROJECT_ROOT / MODEL_PATH))
    print(f"模型已加载: {MODEL_PATH}")

    total_count = 0

    for video_name, frame_dir in FRAME_DIRS.items():
        if not frame_dir.exists():
            print(f"[SKIP] {frame_dir} 不存在")
            continue

        pos_dir = ANNO_POS_BASE / video_name
        pos_dir.mkdir(parents=True, exist_ok=True)

        imgs = sorted([f for f in frame_dir.iterdir() if f.suffix == ".png"])
        print(f"\n[{video_name}] 共 {len(imgs)} 帧")

        count = 0

        for img_path in imgs:
            # 推理
            results = model(str(img_path), verbose=False)
            result = results[0]
            img_h, img_w = result.orig_shape

            shapes = mask_to_shapes(result)
            anno = make_xanylabeling_json(img_path.name, img_w, img_h, shapes)

            json_name = img_path.stem + ".json"
            out_json = pos_dir / json_name
            out_img = pos_dir / img_path.name

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(anno, f, indent=2, ensure_ascii=False)
            shutil.copy2(img_path, out_img)
            
            count += 1

        print(f"  处理了: {count} 帧 (包含有目标的和无目标的)")
        total_count += count

    print(f"\n{'='*40}")
    print(f"总计: 处理并保存了 {total_count} 帧")
    print("完成!")


if __name__ == "__main__":
    main()

