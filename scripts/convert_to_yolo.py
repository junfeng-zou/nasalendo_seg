import json
import os
import shutil
from pathlib import Path

# ──────── 配置 ────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POS_BASE = PROJECT_ROOT / "annotations" / "positive" / "formal"
NEG_BASE = PROJECT_ROOT / "annotations" / "negative" / "formal"

# 需要转换和分配的视频及其对应的限制条件
TARGETS = {
    "output4": None,  # None 表示转换该目录下所有的
    "output5": None,
    "output6": None,
    "output3": 200    # 表示只处理排序后的前 200 张
}

CLASSES = {"instrument": 0}  # 类别名称映射到 YOLO 类别 ID

def poly2yolo(points, img_w, img_h):
    """
    将 JSON 中的像素坐标 points 转换为 YOLO 的归一化坐标 [x0, y0, x1, y1, ...]
    """
    yolo_points = []
    for x, y in points:
        yolo_points.append(str(round(x / img_w, 6)))
        yolo_points.append(str(round(y / img_h, 6)))
    return " ".join(yolo_points)

def main():
    total_pos = 0
    total_neg = 0

    for video, limit in TARGETS.items():
        pos_dir = POS_BASE / video
        neg_dir = NEG_BASE / video
        neg_dir.mkdir(parents=True, exist_ok=True)

        if not pos_dir.exists():
            print(f"[SKIP] {pos_dir} 不存在")
            continue

        print(f"\n处理视频: {video}")

        # 获取所有要处理的 png 文件列表并排序
        imgs = sorted([f for f in pos_dir.iterdir() if f.suffix == ".png"])

        if limit is not None:
            imgs = imgs[:limit]
            print(f"  限制处理前 {limit} 张, 实际获取 {len(imgs)} 张")
        else:
            print(f"  获取到 {len(imgs)} 张")

        pos_count = 0
        neg_count = 0

        for img_path in imgs:
            json_path = pos_dir / (img_path.stem + ".json")
            txt_path = pos_dir / (img_path.stem + ".txt") # YOLO 标签保存路径

            if not json_path.exists():
                print(f"  [警告] 找不到对应的 JSON 文件: {json_path.name}")
                continue

            has_label = False
            yolo_lines = []

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                img_w = data['imageWidth']
                img_h = data['imageHeight']

                for shape in data['shapes']:
                    label = shape['label']
                    if label in CLASSES:
                        cls_id = CLASSES[label]
                        points_str = poly2yolo(shape['points'], img_w, img_h)
                        yolo_lines.append(f"{cls_id} {points_str}")
                        has_label = True

            # 无论正负样本都需要写 txt 文件 (负样本通常是空 txt)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))

            # 判断负样本并移动
            if not has_label:
                # 移动 png
                dst_img = neg_dir / img_path.name
                shutil.move(str(img_path), str(dst_img))

                # 移动 json
                dst_json = neg_dir / json_path.name
                shutil.move(str(json_path), str(dst_json))

                # 移动 txt
                dst_txt = neg_dir / txt_path.name
                shutil.move(str(txt_path), str(dst_txt))

                neg_count += 1
            else:
                pos_count += 1

        print(f"  完成 -> 正样本保留在 positive: {pos_count}, 负样本移动到 negative: {neg_count}")
        total_pos += pos_count
        total_neg += neg_count

    print(f"\n=======================")
    print(f"总计处理完成! 正样本: {total_pos}, 负样本: {total_neg}")

if __name__ == "__main__":
    main()
