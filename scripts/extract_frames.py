#!/usr/bin/env python3
"""
从内窥镜视频中按指定帧率抽帧。
用法: python scripts/extract_frames.py
"""

import cv2
import os
from pathlib import Path

# ===== 配置 =====
VIDEOS = [
    "videos/output1.avi",
    "videos/output2.avi",
]
OUTPUT_DIR = "frames"
TARGET_FPS = 4  # 每秒抽4帧
# ================

def extract_frames(video_path: str, output_dir: str, target_fps: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps > 0 else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 每隔多少帧取一帧
    frame_interval = max(1, round(src_fps / target_fps))
    expected_count = total_frames // frame_interval

    video_name = Path(video_path).stem
    print(f"📹 {video_path}")
    print(f"   分辨率: {w}x{h}, 原始FPS: {src_fps:.1f}, 总帧数: {total_frames}, 时长: {duration:.1f}s")
    print(f"   抽帧间隔: 每{frame_interval}帧取1帧, 预计抽取: ~{expected_count}帧")

    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            filename = f"{video_name}_{saved:05d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1
            if saved % 100 == 0:
                print(f"   已保存 {saved} 帧...")
        frame_idx += 1

    cap.release()
    print(f"   ✅ 共保存 {saved} 帧")
    return saved


def main():
    print("=" * 50)
    print("🎬 内窥镜视频抽帧 (每秒{}帧)".format(TARGET_FPS))
    print("=" * 50)

    total = 0
    for video in VIDEOS:
        if not os.path.exists(video):
            print(f"❌ 文件不存在: {video}")
            continue
        saved = extract_frames(video, OUTPUT_DIR, TARGET_FPS)
        total += saved
        print()

    print(f"📊 总计保存 {total} 帧到 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
