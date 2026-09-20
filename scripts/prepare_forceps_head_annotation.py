#!/usr/bin/env python3
"""Create a small, independent X-AnyLabeling head-box annotation package."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("TooFar", "Good", "TooClose")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distributed_selection(rows, per_class):
    """Balance videos subject to availability, then spread picks in frame order."""
    selected = []
    for label in CLASSES:
        groups = defaultdict(list)
        for row in rows:
            if row["label"] == label:
                groups[row["video_id"]].append(row)
        videos = sorted(groups)
        if sum(map(len, groups.values())) < per_class:
            raise ValueError(f"Not enough {label} samples for {per_class}")
        quotas = Counter()
        while sum(quotas.values()) < per_class:
            for video in videos:
                if quotas[video] < len(groups[video]):
                    quotas[video] += 1
                    if sum(quotas.values()) == per_class:
                        break
        for video in videos:
            ordered = sorted(groups[video], key=lambda r: (int(r["frame_index"]), r["sample_id"]))
            count = quotas[video]
            for i in range(count):
                selected.append(ordered[((2 * i + 1) * len(ordered)) // (2 * count)])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="annotation_projects/forceps_head_pilot_20260907")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an annotation package: {output}")
    source, csv_hashes = {}, {}
    for split in ("train", "val", "test"):
        path = ROOT / "auto_labeling_project/data/final_dataset" / f"{split}_labels.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            source[split] = list(csv.DictReader(handle))
        csv_hashes[split] = digest(path)
    videos = {split: {r["video_id"] for r in rows} for split, rows in source.items()}
    paths = {split: {str(Path(r["image_path"]).resolve()) for r in rows} for split, rows in source.items()}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if videos[first] & videos[second] or paths[first] & paths[second]:
            raise ValueError(f"Source split overlap: {first}, {second}")
    training = distributed_selection(source["train"], 48)
    validation = distributed_selection(source["val"], 12)
    pilot_ids = {r["sample_id"] for r in distributed_selection(training, 10)}
    planned = []
    for split, rows in (("train", training), ("val", validation)):
        for row in sorted(rows, key=lambda r: (r["video_id"], int(r["frame_index"]))):
            batch = "03_validation" if split == "val" else (
                "01_pilot_train" if row["sample_id"] in pilot_ids else "02_more_train")
            source_path = Path(row["image_path"])
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if Path(row["sample_id"]).name != row["sample_id"]:
                raise ValueError("sample_id must be a safe filename")
            relative = Path(batch) / (row["sample_id"] + source_path.suffix.lower())
            planned.append((split, row, source_path.resolve(), relative))
    if len({str(path) for _, _, path, _ in planned}) != 180:
        raise ValueError("Expected 180 unique source images")
    if len({str(relative) for _, _, _, relative in planned}) != 180:
        raise ValueError("Duplicate destination names")
    output.mkdir(parents=True)
    records = []
    for split, row, source_path, relative in planned:
        destination = output / relative
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source_path, destination)
        checksum = digest(source_path)
        if digest(destination) != checksum:
            raise RuntimeError(f"Copy verification failed: {destination}")
        records.append({"sample_id": row["sample_id"], "original_split": split,
                        "annotation_batch": relative.parent.name,
                        "annotation_image": relative.as_posix(),
                        "expected_annotation_json": relative.with_suffix(".json").as_posix(),
                        "source_image": str(source_path), "video_id": row["video_id"],
                        "frame_index": row["frame_index"], "source_frame_index": row.get("source_frame_index", ""),
                        "time_sec": row.get("time_sec", ""), "distance_label": row["label"],
                        "image_sha256": checksum})
    write_csv(output / "manifest.csv", records)
    write_csv(output / "annotation_status.csv", [
        {"sample_id": r["sample_id"], "annotation_image": r["annotation_image"],
         "status": "pending", "visibility": "", "reason": "", "notes": ""} for r in records])
    (output / "classes.txt").write_text("forceps_head\n", encoding="utf-8")
    counts = {}
    for batch in sorted({r["annotation_batch"] for r in records}):
        rows = [r for r in records if r["annotation_batch"] == batch]
        counts[batch] = {"n": len(rows), "classes": dict(Counter(r["distance_label"] for r in rows)),
                         "videos": dict(Counter(r["video_id"] for r in rows))}
    audit = {"total": len(records), "batch_counts": counts, "source_csv_sha256": csv_hashes,
             "selection": "Per-class budgets, video-balanced quotas capped by availability, midpoint picks from equal index bins in temporal order",
             "source_splits": {s: sorted(v) for s, v in videos.items()},
             "test_images_included": 0, "copied_without_pixel_changes": True,
             "script_sha256": digest(Path(__file__))}
    (output / "selection_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(README, encoding="utf-8")
    print(json.dumps({"output": str(output), **audit}, ensure_ascii=False, indent=2))


README = """# 抓钳头部框：人工标注包

本包包含 180 张原尺寸图像副本，可直接在 X-AnyLabeling 中打开。图像未缩放、未叠加 mask、尖端或预测框；原数据与既有标注不受影响。

## 标注顺序

| 文件夹 | 张数 | 用途 | TooFar / Good / TooClose |
|---|---:|---|---|
| `01_pilot_train` | 30 | 先确认头部边界与标注口径 | 10 / 10 / 10 |
| `02_more_train` | 114 | 口径确定后继续标注 | 38 / 38 / 38 |
| `03_validation` | 36 | 独立验证视频，保留作验证 | 12 / 12 / 12 |

先用 X-AnyLabeling 的“打开目录”选择 `01_pilot_train`。选择普通矩形工具，类别输入 `forceps_head`，将 JSON 保存到当前图片文件夹，与图像同名。默认保存位置可能受你原来的软件设置影响，标完第一张后确认文件是否落在这里。第一批完成后，可将本包目录交给后续程序读取，无需先导出 YOLO。

三个文件夹互不重复。前两个文件夹都属于原训练划分，可合并用于训练；验证图来自原验证视频 bend_data1。没有抽取测试图。

## 统一标注规则

1. 一个矩形包住目标抓钳的两个钳瓣，从尖端到钳瓣根部／可见铰接处；不把长杆算入头部。
2. 框尽量贴合可确认的完整头部范围，不主动加宽边距。输入裁剪的上下文余量由后续程序统一添加。
3. 张开时包含两个钳瓣。以抓钳本体确定框，不因棉片或组织而扩大边界；矩形内包含少量背景不可避免。
4. 第一轮只有在头部完整范围可可靠判断时才用于尺度实验；不要把遮挡后剩下的可见碎片当作完整头部框，也不要推测严重遮挡部分的边界。
5. 严重遮挡、出画、目标不明确等情况，在 `annotation_status.csv` 将 status 记为 `unjudgeable`，说明原因，暂时跳过。空 JSON 或没有 JSON 不等于“画面没有头部”，不能直接导出成负样本。
6. 已完成且范围可靠时可将 status 记为 `annotated`；未处理保持 `pending`。visibility 可记录 `clear` 或 `partially_occluded`，reason/notes 可用中文。状态表用于区分未标与不可判，不替代 JSON 框坐标。
7. 标注类别只有 `forceps_head`。距离标签已保存在独立清单中，不要将 TooFar、Good、TooClose 作为头部类别。

## 抽样依据与限制

原训练集抽 144 张、原验证集抽 36 张；分别平衡三个距离类别，并尽量分散到每个视频和同类帧序列的不同位置。使用等数量区间的中点抽样，是确定性的时序分散抽样，不是随机样本或完整视频连续帧。

straight_data5 的训练标签里只有 1 张 TooFar，该帧已纳入；该类别的其余名额分配到其他训练视频。因此按视频统计的数量并非完全一致。没有依据模型预测、ROI 是否通过门控或人工可见性筛选，可能包含无法判断的帧。

样本类别比例经过人为平衡，验证指标不代表原始视频的自然类别分布。包中未保证覆盖所有张合、旋转或遮挡状态，第一批标注时需要同时检查这些覆盖情况。

## 文件说明

- `manifest.csv`：每张副本与原图、原划分、视频、帧序号、距离标签、预期 JSON 路径及 SHA256 的对应关系。
- `annotation_status.csv`：人工记录已标、未标或无法判断及原因。
- `classes.txt`：唯一检测类别 `forceps_head`。
- `selection_audit.json`：分批数量、原 CSV 指纹及抽样方法。

文件名包含视频名称，避免不同视频的同名帧冲突。不要修改图像大小或重命名；软件视窗里的放大显示不影响原图坐标，可以正常使用。后续训练应继续维持原来的视频划分，不能将验证图混入训练。

创建脚本：`scripts/prepare_forceps_head_annotation.py`。脚本拒绝覆盖已有目录，以保留人工标注进度。
"""


if __name__ == "__main__":
    main()
