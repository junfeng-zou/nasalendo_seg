#!/usr/bin/env python3
"""
视频推理分割脚本：使用训练好的 YOLO11-seg 模型对视频进行逐帧分割推理，
并将带有分割 overlay 的结果保存为视频文件。

用法:
    python inference/video_segment.py                                    # 默认推理 output5.avi
    python inference/video_segment.py --source videos/output3.mp4        # 指定视频
    python inference/video_segment.py --conf 0.5                         # 调整置信度
    python inference/video_segment.py --no-save-video --save-frames      # 只保存逐帧图片
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO11-seg 视频推理分割")
    parser.add_argument("--source", type=str, default="videos/output5.avi",
                        help="输入视频路径 (默认: videos/output5.avi)")
    parser.add_argument("--model", type=str,
                        default="runs/segment/runs/yolo11s_seg_formal/weights/best.pt",
                        help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU 阈值 (默认: 0.7)")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="推理图像尺寸 (默认: 1024)")
    parser.add_argument("--device", type=str, default="0",
                        help="推理设备 (默认: 0)")
    parser.add_argument("--save-frames", action="store_true",
                        help="同时保存逐帧分割结果图片")
    parser.add_argument("--no-save-video", action="store_true",
                        help="不保存结果视频 (仅保存帧)")
    parser.add_argument("--show", action="store_true",
                        help="实时显示推理结果窗口")
    args = parser.parse_args()

    # ── 路径设置 ──
    project_root = Path(__file__).resolve().parent.parent
    source_path = project_root / args.source
    model_path = project_root / args.model

    video_stem = source_path.stem  # e.g. "output5"
    results_dir = project_root / "results" / video_stem
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.save_frames:
        frames_dir = results_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载模型 ──
    print("=" * 60)
    print("🎬 YOLO11-seg 视频推理分割")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"视频: {args.source}")
    print(f"置信度: {args.conf}, IoU: {args.iou}, ImgSz: {args.imgsz}")
    print(f"输出目录: {results_dir}")
    print("=" * 60)

    model = YOLO(str(model_path))

    # ── 打开视频 ──
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {source_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n视频信息: {w}x{h} @ {fps:.1f} FPS, 共 {total_frames} 帧\n")

    # ── 视频写入器 ──
    writer = None
    if not args.no_save_video:
        out_video_path = results_dir / f"{video_stem}_segmented.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
        print(f"📹 输出视频: {out_video_path} (FPS={fps:.1f})\n")

    # ── 逐帧推理 ──
    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        results = model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                        device=args.device, verbose=False)
        result = results[0]

        # 绘制分割结果 (带 mask overlay)
        annotated = result.plot(
            line_width=2,
            font_size=0.6,
        )

        # 保存视频帧
        if writer is not None:
            writer.write(annotated)

        # 保存单帧图片
        if args.save_frames:
            cv2.imwrite(str(frames_dir / f"{video_stem}_{frame_idx:05d}.png"), annotated)

        # 实时显示
        if args.show:
            cv2.imshow("YOLO11-seg Inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⚠️  用户中断")
                break

        # 进度
        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t_start
            speed = frame_idx / elapsed
            eta = (total_frames - frame_idx) / speed if speed > 0 else 0
            print(f"  进度: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) "
                  f"| {speed:.1f} FPS | ETA: {eta:.0f}s")

    # ── 清理 ──
    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"✅ 推理完成!")
    print(f"{'='*60}")
    print(f"处理帧数: {frame_idx}")
    print(f"总耗时: {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")
    if writer is not None:
        print(f"结果视频: {results_dir / f'{video_stem}_segmented.avi'}")
    if args.save_frames:
        print(f"逐帧图片: {frames_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
