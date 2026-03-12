#!/usr/bin/env python3
"""
计算内窥镜视野 (FOV) 的独立脚本。

功能：
1. 从指定的视频或摄像头寻找一帧有效的明亮画面。
2. 提取出亮区（实际有像素信息的区域、非全黑盲区）的掩码。
3. 计算该 FOV 的实际几何特征：
   - 使用最小外接圆得到大圆真正的圆心 (cx, cy) 和 半径 (r)。
4. 将 FOV 中心点和半径等参数导出到 JSON 文件，供其他代码进行裁剪对齐或视场几何参考。
5. 保存一张带有提取出的辅助圆线的可视化参考图，供人工复核。

用法:
    python inference/calculate_fov.py --video path/to/video.mp4
    python inference/calculate_fov.py --camera 0
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="计算内窥镜视野(FOV)几何中心与形状并导出配置")
    parser.add_argument("--video", type=str, default=None,
                        help="视频文件路径；若指定，则从视频提取图像")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头/采集卡编号 (默认: 0)")
    parser.add_argument("--n-frames", type=int, default=300,
                        help="采集多少帧进行统计计算 (默认: 30)")
    parser.add_argument("--dark-threshold", type=int, default=10,
                        help="区分有效像素亮区和黑边盲区的灰度阈值 (默认: 10)")
    parser.add_argument("--out-json", type=str, default="results/fov_config.json",
                        help="输出配置文件的保存路径 (默认: results/fov_config.json)")
    parser.add_argument("--out-img", type=str, default="results/fov_visualization.jpg",
                        help="输出可视化参考图的保存路径 (默认: results/fov_visualization.jpg)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # 1. 打开数据源
    is_video = args.video is not None
    if is_video:
        vid_path = Path(args.video)
        if not vid_path.is_absolute():
            vid_path = project_root / vid_path
        cap = cv2.VideoCapture(str(vid_path))
        print(f"打开视频文件: {vid_path}")
    else:
        cap = cv2.VideoCapture(args.camera)
        # 尝试设定为通常的高清采集规格
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print(f"打开摄像头: /dev/video{args.camera}")

    if not cap.isOpened():
        print("❌ 无法打开视频源，退出。")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"输入分辨率: {width}x{height}")

    # 2. 从第 20 帧开始提取，避开视频前几十帧可能的镜头不稳定或录像起步晃动
    skip_frames = 20
    print(f"正在丢弃前 {skip_frames} 帧，并读取后续帧以计算真实视场...")
    
    # 丢掉前 20 帧
    for _ in range(skip_frames):
        cap.read()

    accumulator = np.zeros((height, width), dtype=np.uint32)
    collected = 0
    sample_frame = None
    single_frame_mask = None
    target_frames = args.n_frames - skip_frames

    while collected < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_mask = (current_gray > args.dark_threshold)
        accumulator += current_mask.astype(np.uint32)
        
        # 挑选稳定范围中间的一帧作为单帧基准，以及最终对比背景图
        if collected == target_frames // 2:
            sample_frame = frame.copy()
            single_frame_mask = np.where(current_mask, np.uint8(255), np.uint8(0))
            
        collected += 1

        if not is_video:
            time.sleep(0.01)

    cap.release()

    if collected == 0:
        print("❌ 无法读取到足够的帧数据，退出。")
        return

    if sample_frame is None:
        # 如果读取的帧数过少未能到达中间，就拿最后一帧当单帧基准
        sample_frame = frame.copy()
        single_frame_mask = np.where(current_mask, np.uint8(255), np.uint8(0))

    # A. 平均视场掩码（出现时间超过一半的像素都不是黑的）
    avg_fov_mask = np.where(accumulator >= max(1, collected // 2), np.uint8(255), np.uint8(0))

    # 去除噪点补全孔洞，使形状平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    avg_fov_mask = cv2.morphologyEx(avg_fov_mask, cv2.MORPH_CLOSE, kernel)
    single_frame_mask = cv2.morphologyEx(single_frame_mask, cv2.MORPH_CLOSE, kernel)

    # 提取几何属性的辅助函数
    def get_enclosing_circle(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        return float(cx), float(cy), float(radius)

    # 3. 计算双重 FOV 视野的大圆几何属性
    avg_result = get_enclosing_circle(avg_fov_mask)
    single_result = get_enclosing_circle(single_frame_mask)
    
    if not avg_result or not single_result:
        print("❌ 提取轮廓失败，未在视频中找到内窥镜视野(可能全部为黑)。")
        return

    avg_cx, avg_cy, avg_radius = avg_result
    sgl_cx, sgl_cy, sgl_radius = single_result

    print("\n✅ FOV 视场几何计算完成:")
    print("  【单帧测定】(当前显示帧):")
    print(f"      圆心: ({sgl_cx:.1f}, {sgl_cy:.1f}), 半径: {sgl_radius:.1f}")
    print("  【平均测定】(多帧聚合):")
    print(f"      圆心: ({avg_cx:.1f}, {avg_cy:.1f}), 半径: {avg_radius:.1f}")

    # 4. 导出为 JSON
    out_json_path = project_root / args.out_json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {
        "frame_width": width,
        "frame_height": height,
        "single_frame": {
            "circle_center_x": sgl_cx,
            "circle_center_y": sgl_cy,
            "circle_radius": sgl_radius
        },
        "average_frames": {
            "circle_center_x": avg_cx,
            "circle_center_y": avg_cy,
            "circle_radius": avg_radius
        }
    }

    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
    print(f"\n💾 数据已导出至: {out_json_path}")

    # 5. 画出可视化的参考线保存成图供人类核对
    if sample_frame is not None:
        vis_img = sample_frame.copy()

        # 画出单帧的最终测量结果（绿色实线环）
        cv2.circle(vis_img, (int(sgl_cx), int(sgl_cy)), int(sgl_radius), (0, 255, 0), 2)
        cv2.drawMarker(vis_img, (int(sgl_cx), int(sgl_cy)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(vis_img, "Single Frame", (int(sgl_cx)+15, int(sgl_cy)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 把平均计算出的"内窥镜大圆"给画出来（红色虚/细线作比对）
        cv2.circle(vis_img, (int(avg_cx), int(avg_cy)), int(avg_radius), (0, 0, 255), 2)
        cv2.drawMarker(vis_img, (int(avg_cx), int(avg_cy)), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(vis_img, "Averaged", (int(avg_cx)+15, int(avg_cy)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out_img_path = project_root / args.out_img
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_path), vis_img)
        print(f"💾 可视化复核用图像已保存至: {out_img_path}")


if __name__ == "__main__":
    main()
