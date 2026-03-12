#!/usr/bin/env python3
"""
实时内窥镜分割脚本：从采集卡/摄像头实时获取内窥镜视频流，
使用训练好的 YOLO11-seg 模型进行实时分割并显示。

后处理流程:
  1. 面积过滤: 剔除像素面积过小的噪点掩码
  2. 形态学开运算: 消除掩码边缘毛刺，使轮廓更平滑
  3. 反光过滤: 剔除掩码区域内平均亮度过高或饱和度过低的检测（强反光易被误检为器械）
  4. 时序过滤: 仅当某区域连续 N 帧均被检测到（且置信度≥阈值）时才视为器械，抑制闪烁误检

用法:
    python inference/realtime_segment.py                     # 默认摄像头 /dev/video0
    python inference/realtime_segment.py --video path.mp4    # 以视频文件为输入源
    python inference/realtime_segment.py --camera 2          # 指定摄像头编号
    python inference/realtime_segment.py --record            # 同时录制结果视频
    python inference/realtime_segment.py --conf 0.5         # 调整置信度
    按 'q' 退出, 按 'r' 开始/停止录制, 按 's' 截图 (视频模式下播完自动结束)
"""

import argparse
import shutil
import time
import json
import zmq
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

class ZMQInfoPublisher:
    """ZMQ Publisher for sending instrument tip and width information."""
    def __init__(self, port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.address = f"tcp://*:{port}"
        try:
            self.socket.bind(self.address)
            print(f"✅ ZMQ Info Publisher listening on {self.address} (topic: INSTRUMENT_INFO)")
        except Exception as e:
            print(f"❌ Failed to bind ZMQ Info Publisher to {self.address}: {e}")
            
    def send_info(self, tip_x, tip_y, width, area):
        payload = {
            "tip": {
                "x": float(tip_x),
                "y": float(tip_y)
            },
            "width": float(width),
            "area": float(area),
            "timestamp": time.time()
        }
        try:
            # 采用 "主题 数据" 格式进行 PUB/SUB 广播
            msg_str = json.dumps(payload, ensure_ascii=False)
            self.socket.send_string(f"INSTRUMENT_INFO {msg_str}")
            return True
        except Exception as e:
            pass
        return False
        
    def close(self):
        try:
            self.socket.close()
            self.context.term()
        except:
            pass

# ── 掩码颜色 (BGR) ──
MASK_COLOR = (0, 200, 0)
MASK_ALPHA = 0.45
CONTOUR_COLOR = (0, 255, 0)
CONTOUR_THICKNESS = 2

# 时序匹配时 bbox IoU 低于此值不视为同一目标
TRACK_IOU_THRESHOLD = 0.3


def _box_iou(box_a, box_b):
    """计算两个 bbox (x1,y1,x2,y2) 的 IoU。"""
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _find_base_and_tip_from_boundary(contour_pts_xy, fov_cx, fov_cy, fov_r, img_h, img_w, boundary_thickness=30):
    """
    通过「轮廓点与 FOV 边界距离」确定器械基点和尖端。
    
    【优化版】不再生成全分辨率圆环掩码，而是直接计算轮廓点到 FOV 圆心的距离，
    筛选出落在边界环带内的轮廓点作为"接触带"。

    Args:
        contour_pts_xy: 轮廓点坐标数组，shape=(N, 2)，列序 x, y (float64)
        fov_cx, fov_cy: FOV 圆心坐标
        fov_r:          FOV 半径（像素），<=0 时退化为矩形边界
        img_h, img_w:   图像高度/宽度（仅矩形退化时使用）
        boundary_thickness: 边界环宽度（像素，默认 30）

    Returns:
        (base_xy, tip_xy)  均为 (int, int)；或 None（接触带不存在时）
    """
    if len(contour_pts_xy) < 5:
        return None

    # ── 1. 筛选落在边界环带内的轮廓点 ──
    if fov_r > 0:
        # 计算每个轮廓点到 FOV 圆心的距离
        dx = contour_pts_xy[:, 0] - fov_cx
        dy = contour_pts_xy[:, 1] - fov_cy
        dists_to_center = np.sqrt(dx * dx + dy * dy)
        r_inner = max(0.0, fov_r - boundary_thickness)
        contact_mask = (dists_to_center >= r_inner) & (dists_to_center <= fov_r)
    else:
        # 退化：靠近图像矩形边框的轮廓点
        t = boundary_thickness
        xs, ys = contour_pts_xy[:, 0], contour_pts_xy[:, 1]
        contact_mask = (xs < t) | (xs >= img_w - t) | (ys < t) | (ys >= img_h - t)

    contact_pts = contour_pts_xy[contact_mask]
    if len(contact_pts) < 3:
        return None   # 器械未与边界接触

    # ── 2. 基准线拟合与基点计算 ──
    contact_pts_f32 = contact_pts.astype(np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(contact_pts_f32, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx.item()), float(vy.item()), float(x0.item()), float(y0.item())
    
    # 直线一般式 Ax + By + C = 0
    A = -vy
    B = vx
    C = vy * x0 - vx * y0
    
    # 基点取接触带质心
    base = contact_pts.mean(axis=0)

    # ── 3. 尖端 = 轮廓上距拟合直线垂直距离最远的点 ──
    dists = np.abs(A * contour_pts_xy[:, 0] + B * contour_pts_xy[:, 1] + C)
    tip = contour_pts_xy[int(np.argmax(dists))]

    def to_int(p):
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    return to_int(base), to_int(tip)



def extract_instrument_features(mask, frame_shape, fov_cx=967.7, fov_cy=529.3, fov_radius=0.0, boundary_thickness=30):
    """
    对单个器械掩码提取 IBVS 用的特征点及器械状态。

    【优化版】使用轮廓点（几百个）替代全前景像素（几万个），
    PCA 和宽度计算均在轮廓点上完成，性能提升 10-50 倍。

    Args:
        mask:               uint8 二值掩码，shape=(H,W)
        frame_shape:        原图 shape，即 (H, W, ...)
        fov_cx:             FOV 圆形横坐标
        fov_cy:             FOV 圆形纵坐标
        fov_radius:         FOV 圆形半径（像素），0 表示退化为矩形边框
        boundary_thickness: 边界检测圆环厚度（像素，默认 30）

    Returns:
        dict 或 None（掩码点数不足时返回 None）
    """
    fh, fw = frame_shape[:2]
    # 标定的内窥镜视野圆心
    img_cx, img_cy = fov_cx, fov_cy

    # ── 提取轮廓点替代全前景像素 ──
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    pts = contour[:, 0, :].astype(np.float64)  # (N, 2)，列序 x, y
    if len(pts) < 10:
        return None

    # ── PCA 已禁用，完全使用 boundary 模式 ──
    # # ── PCA：在轮廓点上做协方差矩阵特征值分解（几百个点，极快）──
    # center = pts.mean(axis=0)
    # pts_centered = pts - center
    # cov = np.cov(pts_centered.T)
    # eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # order = np.argsort(eigenvalues)[::-1]
    # eigenvalues = eigenvalues[order]
    # eigenvectors = eigenvectors[:, order]
    # principal_dir  = eigenvectors[:, 0]
    # transverse_dir = eigenvectors[:, 1]
    # if principal_dir[1] > 0:
    #     principal_dir = -principal_dir
    # if transverse_dir[0] < 0:
    #     transverse_dir = -transverse_dir

    lambda_1 = 0.0
    lambda_2 = 0.0
    center = pts.mean(axis=0)

    # 强制使用 boundary 模式
    boundary_result = _find_base_and_tip_from_boundary(
        pts, img_cx, img_cy, float(fov_radius), fh, fw,
        boundary_thickness=boundary_thickness)
    if boundary_result is None:
        return None

    base_pt, tip_pt = boundary_result
    axis_method = "boundary"

    center_np = np.array(center)
    tip_np = np.array(tip_pt)

    if np.linalg.norm(tip_np - center_np) > 1e-3:
        new_principal = center_np - tip_np
        new_principal = new_principal / np.linalg.norm(new_principal)
        principal_dir = new_principal
        transverse_dir = np.array([-principal_dir[1], principal_dir[0]])

        tip_dist = np.linalg.norm(center_np - tip_np)
        tail_np = center_np + principal_dir * tip_dist
        def _to_int(p):
            return (int(round(float(p[0]))), int(round(float(p[1]))))
        base_pt = _to_int(tail_np)
    else:
        principal_dir = np.array([0.0, -1.0])
        transverse_dir = np.array([1.0, 0.0])

    # ── 投影（在轮廓点上做，仅几百个点）──
    pts_centered = pts - center
    proj_long  = pts_centered @ principal_dir
    proj_short = pts_centered @ transverse_dir
    long_max = float(proj_long.max())
    long_min = float(proj_long.min())

    # PCA 回退分支已禁用
    # if axis_method == "pca":
    #     P1 = center + long_max * principal_dir
    #     P2 = center + long_min * principal_dir
    #     img_center = np.array([img_cx, img_cy])
    #     dist1 = np.linalg.norm(P1 - img_center)
    #     dist2 = np.linalg.norm(P2 - img_center)
    #     if dist1 <= dist2:
    #         tip_pt_np, base_pt_np = P1, P2
    #     else:
    #         tip_pt_np, base_pt_np = P2, P1
    #     def _to_int(p):
    #         return (int(round(float(p[0]))), int(round(float(p[1]))))
    #     tip_pt  = _to_int(tip_pt_np)
    #     base_pt = _to_int(base_pt_np)

    half_long  = max(abs(long_max), abs(long_min))
    half_short = 0.0

    # ── 向量化 bin 分割：用 np.digitize 一次性分配，替代 80 次循环 ──
    tip_proj = np.dot(np.array(tip_pt) - center, principal_dir)
    if abs(tip_proj - long_max) < abs(tip_proj - long_min):
        start_proj = long_max
        end_proj = long_min
    else:
        start_proj = long_min
        end_proj = long_max

    N_BINS = 120
    bin_edges = np.linspace(start_proj, end_proj, N_BINS + 1)

    # np.digitize 一次性将所有轮廓点分配到 bin 中
    # 确保 bin_edges 单调递增
    sorted_edges = np.sort(bin_edges)
    bin_indices = np.digitize(proj_long, sorted_edges) - 1  # 0-based bin index
    bin_indices = np.clip(bin_indices, 0, N_BINS - 1)

    all_widths = []
    all_segments_pts = []

    for b in range(N_BINS):
        in_bin = (bin_indices == b)
        count = in_bin.sum()
        if count < 2:
            continue
        s = proj_short[in_bin]
        w_max, w_min = float(s.max()), float(s.min())
        all_widths.append(w_max - w_min)

        pts_in_bin = pts[in_bin]
        idx_max = np.argmax(s)
        idx_min = np.argmin(s)
        all_segments_pts.append((pts_in_bin[idx_max], pts_in_bin[idx_min]))

    # 计算辅助深度特征
    tip_np = np.array(tip_pt)
    base_np = np.array(base_pt) if base_pt is not None else center
    pixel_length = float(np.linalg.norm(tip_np - base_np))

    if len(all_widths) > 0:
        total_valid_bins = len(all_widths)
        start_bin = int(total_valid_bins * 0.01)
        end_bin = int(total_valid_bins * 0.3)

        if end_bin <= start_bin:
            end_bin = start_bin + 3
        end_bin = min(end_bin, total_valid_bins)

        if start_bin >= total_valid_bins:
            start_bin = 0
            end_bin = total_valid_bins

        target_widths = all_widths[start_bin:end_bin]
        avg_width_px = float(np.mean(target_widths)) if target_widths else 0.0
        width_segments_pts = all_segments_pts[start_bin:end_bin]

        if not width_segments_pts:
            width_segments_pts = [all_segments_pts[0]]
    else:
        avg_width_px = 0.0
        width_segments_pts = []

    def to_int_pt(p):
        return (int(round(float(p[0]))), int(round(float(p[1]))))

    P1_pca = center + long_max * principal_dir
    P2_pca = center + long_min * principal_dir

    return {
        "center":         to_int_pt(center),
        "tip":            tip_pt,         # 更新为边界提取的 tip 或 PCA tip
        "tail":           base_pt,        # 对于 boundary 也就是 base，对于 pc 是 tail
        "base":           base_pt,        # 始终包含 base，用于扩展绘制
        "axis_method":    axis_method,
        "points":         [to_int_pt(P1_pca), to_int_pt(P2_pca)],
        "width_segments": width_segments_pts, # 新增用于可视化的测量截面点对
        "principal_dir":  principal_dir,
        "transverse_dir": transverse_dir,
        "half_long":      half_long,
        "half_short":     half_short,
        "avg_width_px":   avg_width_px,
        "pixel_length":   pixel_length,   # 新增：器械画面像素长度
        "lambda_1":       lambda_1,
    }

def _draw_features(annotated, features):
    """将特征提取结果绘制到画面上（主轴、横轴、基点/尖端及模式标签）。"""
    if features is None:
        return

    tip            = features["tip"]
    tail           = features["tail"]
    center         = features["center"]
    points         = features["points"]
    avg_width_px   = features["avg_width_px"]
    pixel_length   = features.get("pixel_length", 0.0) # 新增长度
    axis_method    = features.get("axis_method", "pca")
    principal_dir  = features["principal_dir"]
    transverse_dir = features["transverse_dir"]
    half_long      = features["half_long"]
    half_short     = features["half_short"]

    # 颜色方案
    COLOR_TIP_PT    = (255,   0, 255)   # 品红：尖端点
    COLOR_TAIL_PT   = (0,   200, 255)   # 青：末端点
    COLOR_BASE_PT   = (0,   165, 255)   # 橙：基点（boundary 模式）
    COLOR_SIDE_PT   = (255, 200,   0)   # 金：横轴点
    COLOR_AXIS_L    = (50,  255,  50)   # 亮绿：主轴箭头
    COLOR_AXIS_T    = (50,  200, 255)   # 青：横轴箭头

    cx, cy = center
    
    # ── 绘制主轴 ──
    # 主轴的箭头指向 tip 和 tail
    cv2.arrowedLine(annotated, center, tip,  COLOR_AXIS_L, 2, tipLength=0.08)
    cv2.arrowedLine(annotated, center, tail, COLOR_AXIS_L, 2, tipLength=0.06)

    P1, P2 = points

    # ── 绘制4个特征点 ──
    # P1/P2 在 PCA 主轴上，仍然标出
    cv2.circle(annotated, P1, 7, COLOR_TIP_PT,  -1)
    cv2.circle(annotated, P1, 7, (0, 0, 0), 1)
    cv2.circle(annotated, P2, 7, COLOR_TAIL_PT, -1)

    # ── 绘制用于测量宽度的区间横截面 (暂时隐藏) ──
    # if "width_segments" in features:
    #     for pt1, pt2 in features["width_segments"]:
    #         p1_int = (int(round(float(pt1[0]))), int(round(float(pt1[1]))))
    #         p2_int = (int(round(float(pt2[0]))), int(round(float(pt2[1]))))
    #         # 使用醒目的品红色/黄色画出实际测量的截面线
    #         cv2.line(annotated, p1_int, p2_int, (0, 255, 255), 2)

    if axis_method == "boundary":
        # ── boundary 模式：高亮尖端（品红），不画 base ──
        cv2.circle(annotated, tip, 11, COLOR_TIP_PT, -1)
        cv2.circle(annotated, tip, 11, (255, 255, 255), 2)
        cv2.putText(annotated, "TIP", (tip[0] + 13, tip[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TIP_PT, 2)
    else:
        # ── PCA 备用模式：仅标记尖端 ──
        cv2.circle(annotated, tip, 10, COLOR_TIP_PT, -1)
        cv2.circle(annotated, tip, 10, (255, 255, 255), 2)
        cv2.putText(annotated, "TIP", (tip[0] + 13, tip[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TIP_PT, 2)

    # ── 中心点 ──
    cv2.drawMarker(annotated, center, (0, 255, 0), cv2.MARKER_CROSS, 16, 2)

    # ── 平均宽度标签：显示在被测量的截面中间 ──
    width_text = f"W: {avg_width_px:.1f}px"
    if "width_segments" in features and len(features["width_segments"]) > 0:
        mid_idx = len(features["width_segments"]) // 2
        mid_pt1, mid_pt2 = features["width_segments"][mid_idx]
        text_origin = (mid_pt1 + mid_pt2) / 2.0
        lx = int(np.clip(text_origin[0] + 15, 5, annotated.shape[1] - 120))
        ly = int(np.clip(text_origin[1] - 8,  12, annotated.shape[0] - 5))
    else:
        lx = int(np.clip(center[0] + 10, 5, annotated.shape[1] - 120))
        ly = int(np.clip(center[1] - 8,  12, annotated.shape[0] - 5))
        
    cv2.putText(annotated, width_text, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    # ── 轴向确定模式标签（右上角附近 tip 点）──
    method_text  = "[boundary]" if axis_method == "boundary" else "[PCA]"
    method_color = COLOR_BASE_PT if axis_method == "boundary" else COLOR_TIP_PT
    cv2.putText(annotated, method_text, (tip[0] + 8, tip[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, method_color, 2)


def postprocess_and_draw(frame, result, min_area_ratio=0.005, morph_kernel_size=5,
                         max_brightness=245, min_saturation=15,
                         track_state=None, min_consecutive_frames=5,
                         fov_cx=967.7, fov_cy=529.3, fov_radius=0.0):
    """
    对 YOLO 分割结果进行后处理并绘制到画面上。
    
    后处理步骤:
      1. 面积过滤: 掩码像素面积 < min_area_ratio * 图像面积 的检测被剔除
      2. 形态学开运算: 先腐蚀后膨胀，消除边缘毛刺
      3. 反光过滤: 掩码区域平均亮度过高或饱和度过低则剔除（强反光易被误检为器械）
      4. 时序过滤: 若提供 track_state，仅绘制连续 min_consecutive_frames 帧均被检测到的区域
      5. 特征提取: 椭圆拟合、4点提取、尖端识别、器械状态判断
    
    Args:
        frame: 原始帧 (BGR)
        result: YOLO 推理结果
        min_area_ratio: 最小面积比例阈值 (相对于图像总面积)
        morph_kernel_size: 形态学核大小
        max_brightness: 掩码区域平均亮度上限，超过则视为反光剔除 (0–255，默认 245)
        min_saturation: 掩码区域平均饱和度下限，低于则视为过曝/反光剔除 (0–255，默认 15)
        track_state: 跨帧轨迹状态 dict，含 'tracks': [{'bbox', 'count'}, ...]，每帧传入并会被原地更新
        min_consecutive_frames: 时序过滤所需连续帧数，≤1 时不启用时序过滤
        fov_cx: FOV 圆心横坐标
        fov_cy: FOV 圆心纵坐标
        fov_radius: FOV 圆形半径
    
    Returns:
        annotated:    叠加了后处理掩码与特征点的帧
        num_valid:    经过过滤后的有效检测数量
        features_list: 每个有效器械对应的 extract_instrument_features 返回 dict 列表
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    img_area = h * w
    min_area = int(img_area * min_area_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                       (morph_kernel_size, morph_kernel_size))
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    use_temporal = (track_state is not None and min_consecutive_frames > 1)
    if use_temporal and "tracks" not in track_state:
        track_state["tracks"] = []

    if result.masks is None or len(result.masks) == 0:
        if use_temporal:
            for t in track_state["tracks"]:
                t["count"] = 0
        return annotated, 0, []

    # 获取所有 box（与 mask 索引对应）
    boxes_xyxy = None
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy
        boxes_xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.array(xyxy)

    # ── 第一遍：收集通过面积+形态学+反光过滤的候选 (bbox, mask, conf, idx) ──
    candidates = []
    for i, mask_data in enumerate(result.masks.data):
        mask = mask_data.cpu().numpy().astype(np.uint8)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        area = cv2.countNonZero(mask)
        if area < min_area:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        area = cv2.countNonZero(mask)
        if area < min_area:
            continue
        # ── 只保留最大连通区域（用轮廓替代 connectedComponentsWithStats，更快）──
        tmp_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(tmp_contours) > 1:
            largest_contour = max(tmp_contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 1, -1)
        area = cv2.countNonZero(mask)
        if area < min_area:
            continue
        mask_bool = mask > 0
        mean_brightness = float(gray[mask_bool].mean())
        mean_saturation = float(hsv[:, :, 1][mask_bool].mean())
        if mean_brightness > max_brightness or mean_saturation < min_saturation:
            continue
        bbox = None
        if boxes_xyxy is not None and i < len(boxes_xyxy):
            bbox = tuple(map(float, boxes_xyxy[i]))
        else:
            xs, ys = np.where(mask > 0)
            if len(xs) > 0:
                x1, x2 = float(ys.min()), float(ys.max())
                y1, y2 = float(xs.min()), float(xs.max())
                bbox = (x1, y1, x2, y2)
        if bbox is None:
            continue
        conf = 0.0
        if result.boxes is not None and i < len(result.boxes):
            conf = float(result.boxes.conf[i])
        candidates.append({"bbox": bbox, "mask": mask, "conf": conf, "idx": i})

    # ── 时序过滤：用 bbox IoU 与上一帧轨迹匹配，仅保留连续出现 ≥ min_consecutive_frames 的候选 ──
    tracks = track_state["tracks"] if use_temporal else []
    if use_temporal and len(candidates) == 0:
        for t in tracks:
            t["count"] = 0
    if use_temporal:
        # 本帧每个 track 最多匹配一个 candidate，每个 candidate 最多匹配一个 track
        assigned_track = [False] * len(tracks)
        assigned_cand = [False] * len(candidates)
        for ti, t in enumerate(tracks):
            best_j, best_iou = -1, TRACK_IOU_THRESHOLD
            for j, c in enumerate(candidates):
                if assigned_cand[j]:
                    continue
                iou = _box_iou(t["bbox"], c["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                assigned_cand[best_j] = True
                assigned_track[ti] = True
                t["bbox"] = candidates[best_j]["bbox"]
                t["count"] = t.get("count", 0) + 1
                candidates[best_j]["track_count"] = t["count"]
            else:
                t["count"] = 0
        for j, c in enumerate(candidates):
            if not assigned_cand[j]:
                c["track_count"] = 1
                tracks.append({"bbox": c["bbox"], "count": 1})
        # 只保留本帧出现且连续帧数达标的候选
        to_draw = [c for c in candidates if c.get("track_count", 1) >= min_consecutive_frames]
        # 未在本帧匹配到的 track 计数清零（已在上面 t["count"] = 0 处理）
        # 轨迹过多时清理，只保留最近有匹配的以控制内存
        if len(tracks) > 50:
            track_state["tracks"] = [t for t in tracks if t.get("count", 0) > 0][:20]
    else:
        to_draw = candidates

    # ── 单一器械约束：只保留置信度最高的一个候选 ──
    if len(to_draw) > 1:
        to_draw = [max(to_draw, key=lambda c: c["conf"])]

    num_valid = len(to_draw)
    features_list = []

    for c in to_draw:
        mask = c["mask"]
        conf = c["conf"]

        # ── 半透明掩码（原地混合，避免创建全分辨率临时数组）──
        mask_region = mask > 0
        annotated[mask_region] = (
            annotated[mask_region].astype(np.float32) * (1.0 - MASK_ALPHA) +
            np.array(MASK_COLOR, dtype=np.float32) * MASK_ALPHA
        ).astype(np.uint8)

        # ── 轮廓线 ──
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)

        # ── 置信度标签 ──
        if contours:
            top_point = min(contours[0], key=lambda p: p[0][1])
            tx, ty = top_point[0]
            label = f"instrument {conf:.2f}"
            cv2.putText(annotated, label, (int(tx), max(int(ty) - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONTOUR_COLOR, 2)
                        
            # 计算当前轮廓的物理面积 (像素数)
            mask_area = cv2.contourArea(contours[0])
        else:
            mask_area = 0.0

        # ── 特征点提取 + 绘制 ──
        features = extract_instrument_features(mask, frame.shape, fov_cx=fov_cx, fov_cy=fov_cy, fov_radius=fov_radius)
        if features is not None:
            features["mask_area"] = mask_area
            _draw_features(annotated, features)
            features_list.append(features)

    # ── 绘制视野参考同心圆 ──
    ref_cx, ref_cy = 967, 529  # 四舍五入后的标定中心
    for r in [300, 400, 500]:
        cv2.circle(annotated, (ref_cx, ref_cy), r, (255, 255, 0), 1, cv2.LINE_AA)
        # 在圆圈旁边加个小文字标注半径
        cv2.putText(annotated, str(r), (ref_cx, ref_cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return annotated, num_valid, features_list


def main():
    parser = argparse.ArgumentParser(description="YOLO11-seg 实时内窥镜分割")
    parser.add_argument("--video", type=str, default=None,
                        help="视频文件路径；指定后以视频为输入源，不再使用摄像头")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头/采集卡编号 (默认: 0)，仅在未指定 --video 时生效")
    parser.add_argument("--model", type=str,
                        default="runs/segment/yolo11s_seg_formal/weights/best.pt",
                        help="模型权重路径 (相对项目根)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU 阈值 (默认: 0.7)")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="推理图像尺寸 (默认: 1024)")
    parser.add_argument("--device", type=str, default="0",
                        help="推理设备 (默认: 0)")
    parser.add_argument("--min-area", type=float, default=0.005,
                        help="最小掩码面积比例 (默认: 0.005, 即图像面积的 0.5%%)")
    parser.add_argument("--morph-kernel", type=int, default=5,
                        help="形态学开运算核大小 (默认: 5)")
    parser.add_argument("--max-brightness", type=int, default=245,
                        help="反光过滤: 掩码区域平均亮度超过此值则剔除 (0–255, 默认 245, 设为 256 关闭)")
    parser.add_argument("--min-saturation", type=int, default=15,
                        help="反光过滤: 掩码区域平均饱和度低于此值则剔除 (0–255, 默认 15, 设为 0 关闭)")
    parser.add_argument("--min-consecutive-frames", type=int, default=2,
                        help="时序过滤: 仅当某区域连续 N 帧均被检测到才视为器械 (默认 5, 设为 1 关闭)")
    parser.add_argument("--record", action="store_true",
                        help="启动时立即开始录制 (以视频为输入时将自动保存)")
    parser.add_argument("--no-display", action="store_true",
                        help="不显示窗口 (仅录制模式)")
    parser.add_argument("--zmq-port", type=int, default=5556,
                        help="ZMQ PUB 发送坐标与宽度的端口 (默认: 5556)")
    args = parser.parse_args()

    # ── 路径设置 ──
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / args.model
    if not model_path.exists():
        print(f"❌ 未找到模型权重: {model_path}")
        print("   请先训练模型 (train.py) 或通过 --model 指定已有权重路径，例如:")
        print("   python inference/realtime_segment.py --model path/to/best.pt")
        return
    results_dir = project_root / "videos" / "depth_explore"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载模型 ──
    print("=" * 60)
    print("🔴 YOLO11-seg 实时内窥镜分割")
    print("=" * 60)
    print(f"模型: {args.model}")
    if args.video:
        print(f"输入: 视频 {args.video}")
    else:
        print(f"输入: 摄像头 /dev/video{args.camera}")
    print(f"置信度: {args.conf}, IoU: {args.iou}, ImgSz: {args.imgsz}")
    print(f"面积过滤: {args.min_area*100:.1f}%, 形态学核: {args.morph_kernel}x{args.morph_kernel}")
    print(f"反光过滤: 亮度<{args.max_brightness}, 饱和度>{args.min_saturation}")
    print(f"时序过滤: 连续 {args.min_consecutive_frames} 帧才显示 (设为 1 关闭)")
    print(f"输出目录: {results_dir}")
    print("=" * 60)

    # ── 初始化 ZMQ Publisher ──
    zmq_publisher = ZMQInfoPublisher(port=args.zmq_port)

    model = YOLO(str(model_path))

    # ── 加载 FOV 配置文件 ──
    fov_cx, fov_cy, fov_radius = 967.7061157226562, 529.3329467773438, 0.0
    fov_config_path = project_root / "results" / "fov_config.json"
    if fov_config_path.exists():
        try:
            with open(fov_config_path, "r", encoding="utf-8") as f:
                fov_data = json.load(f)
                if "average_frames" in fov_data:
                    avg_data = fov_data["average_frames"]
                    fov_cx = avg_data.get("circle_center_x", fov_cx)
                    fov_cy = avg_data.get("circle_center_y", fov_cy)
                    fov_radius = avg_data.get("circle_radius", fov_radius)
            print(f"✅ 成功加载 FOV 配置: 中心 ({fov_cx:.1f}, {fov_cy:.1f}), 半径 {fov_radius:.1f}")
        except Exception as e:
            print(f"⚠️  加载 FOV 配置失败: {e}，将使用默认设置")
    else:
        print(f"⚠️  未找到 FOV 配置文件 ({fov_config_path})，将使用默认设置 (半径 0.0，中心 {fov_cx:.1f}, {fov_cy:.1f})")

    # ── 打开视频源（视频文件或摄像头）──
    is_video_file = args.video is not None and args.video.strip() != ""
    if is_video_file:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = project_root / video_path
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n视频信息: {actual_w}x{actual_h} @ {cam_fps:.1f} FPS, 共 {total_frames} 帧")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头: /dev/video{args.camera}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = None
        print(f"\n摄像头信息: {actual_w}x{actual_h} @ {cam_fps:.1f} FPS")

    # ── 录制器 ──
    writer = None
    is_recording = args.record or is_video_file

    def start_recording():
        nonlocal writer
        if is_video_file:
            input_name = Path(args.video).stem
            out_path = results_dir / f"{input_name}_seg.avi"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = results_dir / f"realtime_{timestamp}.avi"
            
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_path), fourcc, cam_fps, (actual_w, actual_h))
        print(f"\n🔴 开始保存视频: {out_path}")
        return out_path

    def stop_recording():
        nonlocal writer
        if writer is not None:
            writer.release()
            writer = None
            print("⬛ 停止录制")

    if is_recording:
        recording_path = start_recording()

    # ── 主循环 ──
    print("\n快捷键: [q] 退出 | [r] 录制开关 | [s] 截图\n")

    frame_count = 0
    fps_timer = time.time()
    display_fps = 0.0
    # 时序过滤用的跨帧轨迹状态（仅当 min_consecutive_frames > 1 时使用）
    track_state = {} if args.min_consecutive_frames > 1 else None

    try:
        while True:
            # 记录当前帧开始时间以控制播放速度
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    print("\n视频播放完毕")
                    break
                print("⚠️  无法读取帧，尝试重连...")
                time.sleep(0.5)
                continue

            # 推理
            t0 = time.time()
            results = model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                            device=args.device, verbose=False)
            result = results[0]
            t1 = time.time()

            # 后处理 + 绘制 (面积/形态学/反光/时序过滤 + IBVS 特征提取)
            annotated, num_valid, features_list = postprocess_and_draw(
                frame, result,
                min_area_ratio=args.min_area,
                morph_kernel_size=args.morph_kernel,
                max_brightness=args.max_brightness,
                min_saturation=args.min_saturation,
                track_state=track_state,
                min_consecutive_frames=args.min_consecutive_frames,
                fov_cx=fov_cx,
                fov_cy=fov_cy,
                fov_radius=fov_radius
            )
            t2 = time.time()

            # 计算实时 FPS
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # 在画面上叠加状态信息 (注释掉以保证画面干净)
            status_text = f"FPS: {display_fps:.1f}"
            if is_video_file:
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                status_text += f"  Frame: {current_frame}"
                if total_frames and total_frames > 0:
                    status_text += f"/{total_frames}"
            if is_recording:
                # 录制指示红点
                cv2.circle(annotated, (actual_w - 30, 30), 12, (0, 0, 255), -1)

            cv2.putText(annotated, status_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 检测数量 (过滤后)
            det_text = f"Instruments: {num_valid}"
            cv2.putText(annotated, det_text, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # IBVS 特征信息（取第一个器械显示到画面左下角）
            if features_list:
                feat = features_list[0]
                tip = feat["tip"]
                pts = feat["points"]
                l1 = feat.get("lambda_1", 0.0)
                l2 = feat.get("lambda_2", 0.0)
                ratio = (l1 / l2) if l2 > 0 else 0.0
                cv2.putText(annotated, f"PCA L1: {l1:.0f}, L2: {l2:.0f}, L1/L2: {ratio:.2f}", (10, actual_h - 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(annotated, f"TIP: ({tip[0]}, {tip[1]})", (10, actual_h - 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
                
                avg_w = feat.get("avg_width_px", 0.0)
                mask_area = feat.get("mask_area", 0.0)
                
                cv2.putText(annotated, f"Tip Width: {avg_w:.1f} px", (10, actual_h - 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
                            
                cv2.putText(annotated, f"Area: {mask_area:.1f} px^2", (10, actual_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                            
                # ── ZMQ 发送信息 ──
                if 'zmq_publisher' in locals() or 'zmq_publisher' in globals():
                    zmq_publisher.send_info(tip[0], tip[1], avg_w, mask_area)
                
                p_len = feat.get("pixel_length", 0.0)
                cv2.putText(annotated, f"Pixel Length: {p_len:.1f} px", (10, actual_h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2)
                            
                pts_str = "  ".join(f"P{i+1}:({p[0]},{p[1]})" for i, p in enumerate(pts[:2]))
                cv2.putText(annotated, pts_str, (10, actual_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

            # 录制
            t3 = time.time()
            if is_recording and writer is not None:
                writer.write(annotated)
            t4 = time.time()

            # 【性能分析】每 30 帧打印一次各步骤耗时
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
                print(f"[PROFILE] YOLO: {(t1-t0)*1000:.1f}ms | PostProc: {(t2-t1)*1000:.1f}ms | "
                      f"Record: {(t4-t3)*1000:.1f}ms | Total: {(t4-t0)*1000:.1f}ms")

            # 显示和控制播放速度
            if not args.no_display:
                cv2.imshow("Endoscope Segmentation (q=quit, r=record, s=screenshot)", annotated)

                # 动态计算所需的 waitKey 延时，使播放速度逼近原视频 FPS
                process_time = time.time() - frame_start_time
                target_delay = 1.0 / cam_fps
                wait_time_ms = int((target_delay - process_time) * 1000)
                wait_time_ms = max(1, wait_time_ms)  # 至少延时 1ms 以让 GUI 刷新

                key = cv2.waitKey(wait_time_ms) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    # 切换录制状态
                    is_recording = not is_recording
                    if is_recording:
                        recording_path = start_recording()
                    else:
                        stop_recording()
                elif key == ord("s"):
                    # 截图
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = results_dir / f"screenshot_{timestamp}.png"
                    cv2.imwrite(str(screenshot_path), annotated)
                    print(f"📸 截图已保存: {screenshot_path}")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断 (Ctrl+C)")

    # ── 清理 ──
    stop_recording()
    cap.release()
    cv2.destroyAllWindows()
    if 'zmq_publisher' in locals():
        zmq_publisher.close()

    print(f"\n{'='*60}")
    print("✅ 实时分割已结束")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
