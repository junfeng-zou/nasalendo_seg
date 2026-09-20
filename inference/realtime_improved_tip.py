#!/usr/bin/env python3
"""YOLO11-seg 实时器械分割与改进的尖端几何检测。

相比 realtime_segment.py 中的单点最远距离方法，本脚本增加：
1. 使用相对 FOV 圆心的最外侧轮廓簇确定根部，降低固定 FOV 边界漂移的影响。
2. 使用远端候选区域的中心，而不是单个极值像素。
3. 开口器械存在两个远端簇时，输出两簇中点。
4. 径向几何退化时，使用 PCA + 前一帧尖端回退。
5. One Euro Filter、尖端跳变门控和短时丢失保持。
6. 输出观测置信度和当前提取模式。
7. 启动时使用前若干帧自动标定当前 FOV。
8. 在候选选择阶段联合连续高光核心和历史 bbox，拒绝大面积反光误检。

示例：
    python inference/realtime_improved_tip.py \\
      --video videos/surgical_videos/2020-11-26_104423_VID001.mp4 \\
      --output segnext_instrument_seg/outputs/sugical_output/VID001_improved_tip.mp4 \\
      --device 0 --no-display
"""

from __future__ import annotations

import argparse
import json
import math
import select
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import zmq
except ImportError:  # ZMQ 为可选功能
    zmq = None


MASK_COLOR = np.array((0, 200, 0), dtype=np.float32)
MASK_ALPHA = 0.42
RADIAL_EXTREME_COLOR = (255, 0, 0)  # BGR：纯蓝，表示径向距离最高的轮廓点


@dataclass
class TipObservation:
    """单帧尖端几何观测。"""

    tip: np.ndarray
    base: np.ndarray
    direction: np.ndarray
    confidence: float
    mode: str
    distal_centers: list[np.ndarray]
    base_candidates: np.ndarray
    radial_extreme_points: np.ndarray


@dataclass
class TrackedTip:
    """时序滤波后的尖端状态。"""

    tip: np.ndarray
    raw_tip: Optional[np.ndarray]
    base: Optional[np.ndarray]
    confidence: float
    mode: str
    valid: bool


def _alpha(cutoff: float, dt: float) -> float:
    cutoff = max(float(cutoff), 1e-6)
    dt = max(float(dt), 1e-6)
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter2D:
    """适合实时坐标的二维 One Euro Filter。"""

    def __init__(self, min_cutoff=1.2, beta=0.015, derivative_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.raw = None
        self.filtered = None
        self.filtered_derivative = np.zeros(2, dtype=np.float64)

    def reset(self):
        self.raw = None
        self.filtered = None
        self.filtered_derivative[:] = 0.0

    def update(self, point, dt):
        point = np.asarray(point, dtype=np.float64)
        if self.filtered is None:
            self.raw = point.copy()
            self.filtered = point.copy()
            return self.filtered.copy()

        derivative = (point - self.raw) / max(float(dt), 1e-6)
        ad = _alpha(self.derivative_cutoff, dt)
        self.filtered_derivative = (
            ad * derivative + (1.0 - ad) * self.filtered_derivative
        )
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(
            self.filtered_derivative
        )
        a = _alpha(cutoff, dt)
        self.filtered = a * point + (1.0 - a) * self.filtered
        self.raw = point.copy()
        return self.filtered.copy()


class TemporalTipTracker:
    """尖端 One Euro 滤波、跳变拒绝、短时保持与稳定新位置重捕获。"""

    def __init__(
        self,
        min_cutoff=1.2,
        beta=0.015,
        max_jump=120.0,
        hold_frames=3,
        reacquire_frames=3,
        reacquire_radius=80.0,
    ):
        self.filter = OneEuroFilter2D(min_cutoff, beta)
        self.max_jump = float(max_jump)
        self.hold_frames = int(hold_frames)
        self.reacquire_frames = max(1, int(reacquire_frames))
        self.reacquire_radius = float(reacquire_radius)
        self.filtered = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self.base = None
        self.last_confidence = 0.0
        self.missing = 0
        self.pending_tip = None
        self.pending_count = 0

    @property
    def reference_tip(self):
        return None if self.filtered is None else self.filtered.copy()

    @property
    def reference_base(self):
        return None if self.base is None else self.base.copy()

    def reset(self):
        self.filter.reset()
        self.filtered = None
        self.velocity[:] = 0.0
        self.base = None
        self.last_confidence = 0.0
        self.missing = 0
        self._clear_pending()

    def _clear_pending(self):
        self.pending_tip = None
        self.pending_count = 0

    def _accept(self, observation, dt, mode=None, reset_filter=False):
        """接收当前观测；重捕获时会重置旧滤波状态。"""
        raw_tip = observation.tip.astype(np.float64)
        previous = None if self.filtered is None else self.filtered.copy()
        if reset_filter:
            self.filter.reset()
            self.filtered = None
            self.velocity[:] = 0.0
            previous = None

        filtered = self.filter.update(raw_tip, dt)
        if previous is not None:
            displacement = filtered - previous
            self.velocity = 0.75 * self.velocity + 0.25 * displacement

        self.filtered = filtered
        self.base = observation.base.astype(np.float64)
        self.last_confidence = float(observation.confidence)
        self.missing = 0
        self._clear_pending()
        output_mode = mode or observation.mode
        return TrackedTip(
            tip=filtered.copy(),
            raw_tip=raw_tip.copy(),
            base=self.base.copy(),
            confidence=float(observation.confidence),
            mode=output_mode,
            valid=True,
        )

    def _update_pending(self, raw_tip):
        """统计跳变后的新候选是否在连续帧中保持一致。"""
        if (
            self.pending_tip is not None
            and np.linalg.norm(raw_tip - self.pending_tip) <= self.reacquire_radius
        ):
            self.pending_count += 1
        else:
            self.pending_count = 1
        self.pending_tip = raw_tip.copy()
        return self.pending_count

    def _hold(self, reason):
        self.missing += 1
        if self.filtered is None or self.missing > self.hold_frames:
            return None
        confidence = self.last_confidence * (0.55 ** self.missing)
        return TrackedTip(
            tip=self.filtered.copy(),
            raw_tip=None,
            base=None if self.base is None else self.base.copy(),
            confidence=float(confidence),
            mode=reason,
            valid=True,
        )

    def update(self, observation: Optional[TipObservation], dt: float):
        if observation is None:
            # 重捕获必须由连续有效观测构成，中间缺帧则重新计数。
            self._clear_pending()
            return self._hold("held_missing")

        raw_tip = observation.tip.astype(np.float64)
        if self.filtered is not None:
            predicted = self.filtered + self.velocity
            jump = float(np.linalg.norm(raw_tip - predicted))
            # 高置信度观测允许更大位移，但仍拒绝极端跳变。
            allowed_jump = self.max_jump * (1.0 + 0.5 * observation.confidence)
            if jump > allowed_jump:
                pending_count = self._update_pending(raw_tip)
                if pending_count >= self.reacquire_frames:
                    # 新位置连续多帧一致，说明它不是单帧离群点。
                    # 舍弃已过时的旧状态，从新位置重新初始化滤波器。
                    return self._accept(
                        observation,
                        dt,
                        mode=f"reacquired_{observation.mode}",
                        reset_filter=True,
                    )
                return self._hold("held_jump_rejected")
        return self._accept(observation, dt)


class DetectionGate:
    """使用 bbox IoU 进行连续帧确认。"""

    def __init__(self, confirm_frames=2, iou_threshold=0.3):
        self.confirm_frames = max(1, int(confirm_frames))
        self.iou_threshold = float(iou_threshold)
        self.last_bbox = None
        self.hits = 0

    def reset(self):
        self.last_bbox = None
        self.hits = 0

    def update(self, bbox):
        bbox = np.asarray(bbox, dtype=np.float64)
        switched = False
        if self.last_bbox is None:
            self.hits = 1
        elif _box_iou(self.last_bbox, bbox) >= self.iou_threshold:
            self.hits += 1
        else:
            self.hits = 1
            switched = True
        self.last_bbox = bbox
        return self.hits >= self.confirm_frames, switched


class TipPublisher:
    """可选 ZMQ PUB，发布改进后的尖端状态。"""

    def __init__(self, port):
        self.context = None
        self.socket = None
        if port is None:
            return
        if zmq is None:
            print("⚠️  未安装 pyzmq，已关闭 ZMQ 输出")
            return
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{int(port)}")
            print(f"✅ ZMQ 尖端坐标输出: tcp://*:{int(port)}")
        except Exception as exc:
            print(f"⚠️  ZMQ 启动失败，已继续运行: {exc}")
            self.close()

    def send(self, tracked: TrackedTip):
        if self.socket is None or tracked is None or not tracked.valid:
            return
        payload = {
            "tip": {
                "x": float(tracked.tip[0]),
                "y": float(tracked.tip[1]),
            },
            "confidence": float(tracked.confidence),
            "mode": tracked.mode,
            "valid": bool(tracked.valid),
            "timestamp": time.time(),
        }
        self.socket.send_string(
            "INSTRUMENT_TIP " + json.dumps(payload, ensure_ascii=False)
        )

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None


class MatlabTcpPublisher:
    """普通 TCP Server：向 MATLAB 发尖端坐标，并接收其运动方向。"""

    def __init__(self, host="127.0.0.1", port=5555):
        self.host = str(host)
        self.port = None if port is None else int(port)
        self.server = None
        self.client = None
        self.client_address = None
        self.receive_buffer = bytearray()
        self.latest_scope_motion = None
        self.latest_scope_motion_time = None
        if self.port is None:
            return

        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((self.host, self.port))
            self.server.listen(1)
            self.server.setblocking(False)
            print(
                f"✅ MATLAB TCP 尖端坐标输出: {self.host}:{self.port} "
                "(等待 tcpclient 连接)"
            )
        except OSError as exc:
            print(f"⚠️  MATLAB TCP Server 启动失败，已继续运行: {exc}")
            self.close()

    def _close_client(self):
        if self.client is not None:
            try:
                self.client.close()
            except OSError:
                pass
        self.client = None
        self.client_address = None
        self.receive_buffer.clear()
        self.latest_scope_motion = None
        self.latest_scope_motion_time = None

    def _accept_client(self):
        if self.server is None or self.client is not None:
            return
        try:
            client, address = self.server.accept()
        except BlockingIOError:
            return
        except OSError as exc:
            print(f"⚠️  MATLAB TCP 接收连接失败: {exc}")
            return

        client.settimeout(0.02)
        self.client = client
        self.client_address = address
        print(f"✅ MATLAB tcpclient 已连接: {address[0]}:{address[1]}")

    def receive_scope_motion(self, max_age=0.3):
        """非阻塞读取 MATLAB 回传，只保留最新一条有效运动指令。"""
        self._accept_client()
        if self.client is None:
            return None

        try:
            while True:
                readable, _, exceptional = select.select(
                    [self.client], [], [self.client], 0
                )
                if exceptional:
                    self._close_client()
                    return None
                if not readable:
                    break

                chunk = self.client.recv(4096)
                if not chunk:
                    self._close_client()
                    return None
                self.receive_buffer.extend(chunk)
        except (ConnectionResetError, socket.timeout, OSError):
            self._close_client()
            return None

        while b"\n" in self.receive_buffer:
            line, _, remainder = self.receive_buffer.partition(b"\n")
            self.receive_buffer = bytearray(remainder)
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("type") != "scope_motion":
                continue
            try:
                du = float(payload["du"])
                dv = float(payload["dv"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(du) or not math.isfinite(dv):
                continue

            self.latest_scope_motion = {
                "du": du,
                "dv": dv,
                "speed": math.hypot(du, dv),
                "valid": bool(payload.get("valid", True)),
            }
            self.latest_scope_motion_time = time.monotonic()

        if (
            self.latest_scope_motion is None
            or self.latest_scope_motion_time is None
            or time.monotonic() - self.latest_scope_motion_time > float(max_age)
            or not self.latest_scope_motion["valid"]
        ):
            return None
        return self.latest_scope_motion

    def send(self, tracked: TrackedTip, frame_index=None):
        self._accept_client()
        if self.client is None or tracked is None or not tracked.valid:
            return

        payload = {
            "tip": {
                "x": float(tracked.tip[0]),
                "y": float(tracked.tip[1]),
            },
            "confidence": float(tracked.confidence),
            "mode": tracked.mode,
            "valid": bool(tracked.valid),
            "timestamp": time.time(),
        }
        if frame_index is not None:
            payload["frame_index"] = int(frame_index)
        if tracked.raw_tip is not None:
            payload["raw_tip"] = {
                "x": float(tracked.raw_tip[0]),
                "y": float(tracked.raw_tip[1]),
            }

        # MATLAB 按换行切分消息，并把每一整行直接交给 jsondecode。
        message = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self.client.sendall(message)
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError) as exc:
            print(f"⚠️  MATLAB TCP 连接断开，等待重新连接: {exc}")
            self._close_client()

    def close(self):
        self._close_client()
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass
        self.server = None


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _cyclic_true_runs(flags):
    """返回布尔环形序列中的连续 True 索引簇。"""
    flags = np.asarray(flags, dtype=bool)
    n = len(flags)
    if n == 0 or not flags.any():
        return []
    if flags.all():
        return [np.arange(n, dtype=np.int32)]

    false_index = int(np.flatnonzero(~flags)[0])
    start = (false_index + 1) % n
    rotated = np.roll(flags, -start)
    runs = []
    index = 0
    while index < n:
        if not rotated[index]:
            index += 1
            continue
        end = index
        while end < n and rotated[end]:
            end += 1
        runs.append((np.arange(index, end, dtype=np.int32) + start) % n)
        index = end
    return runs


def _largest_clean_component(mask, morph_kernel):
    mask = (mask > 0).astype(np.uint8)
    if morph_kernel > 1:
        size = int(morph_kernel)
        if size % 2 == 0:
            size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [largest], -1, 1, -1)
    return cleaned


def _mask_centroid(mask, contour_points):
    moments = cv2.moments(mask, binaryImage=True)
    if abs(moments["m00"]) > 1e-6:
        return np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float64,
        )
    return contour_points.mean(axis=0)


def _choose_base_runs(points, runs, fov_center, previous_base):
    """选择属于同一外侧根部的连续候选簇。

    上/下裁切形成的直线根部可能在径向候选带中分裂成两个角点簇。
    这里把相对 FOV 圆心方向相近的簇合并，避免 base 在横截面两端跳变。
    """
    # 根部应当具有连续的轮廓支撑；过滤只有少数像素的 FOV 毛刺/尖角。
    min_run_length = max(12, int(round(0.005 * len(points))))
    runs = [run for run in runs if len(run) >= min_run_length]
    if not runs:
        return None

    centers = [np.median(points[run], axis=0) for run in runs]
    directions = []
    for center in centers:
        direction = center - fov_center
        norm = float(np.linalg.norm(direction))
        directions.append(direction / max(norm, 1e-6))

    # 点积 0.5 对应 60 度；同一根部横截面的两个角点通常远小于该夹角。
    unassigned = set(range(len(runs)))
    groups = []
    while unassigned:
        seed = min(unassigned)
        group = [
            index
            for index in sorted(unassigned)
            if float(np.dot(directions[seed], directions[index])) >= 0.5
        ]
        groups.append(group)
        unassigned.difference_update(group)

    group_sizes = [sum(len(runs[index]) for index in group) for group in groups]
    max_size = max(group_sizes)
    # 上一帧 base 只在支撑长度与最大候选接近的簇组之间发挥作用，
    # 防止跟踪状态把一个短小异常簇持续锁定为根部。
    eligible = [
        group
        for group, size in zip(groups, group_sizes)
        if size >= max(min_run_length, int(0.7 * max_size))
    ]

    def group_center(group):
        # 对各连续簇的中心等权平均，避免轮廓采样数量把 base 拉向某一角点。
        return np.mean(np.stack([centers[index] for index in group]), axis=0)

    if previous_base is None:
        selected = max(
            eligible,
            key=lambda group: sum(len(runs[index]) for index in group),
        )
    else:
        previous_base = np.asarray(previous_base, dtype=np.float64)
        selected = min(
            eligible,
            key=lambda group: np.linalg.norm(group_center(group) - previous_base),
        )
    return [runs[index] for index in selected]


def _distal_cluster_centers(points, distal_flags):
    runs = _cyclic_true_runs(distal_flags)
    if not runs:
        return []
    runs = sorted(runs, key=len, reverse=True)
    selected = [runs[0]]
    if len(runs) > 1 and len(runs[1]) >= max(2, int(0.25 * len(runs[0]))):
        selected.append(runs[1])
    return [np.median(points[run], axis=0) for run in selected]


def _radial_outer_tip(
    mask,
    points,
    detection_confidence,
    fov_center,
    base_radial_band,
    radial_extreme_percentile,
    distal_percentile,
    previous_base,
):
    """以相对 FOV 圆心的最外侧连续轮廓簇作为器械根部。

    这里只使用轮廓点的径向排序，不再要求轮廓落入固定 FOV 边界带，
    因而允许真实 FOV 相对标定结果发生小范围平移。使用径向高百分位
    作为稳健外径并向内扩展带状区域，降低少量尖角和轮廓毛刺的影响。
    """
    offsets = points - fov_center
    radial_distances = np.linalg.norm(offsets, axis=1)
    robust_outer_radius = float(
        np.percentile(radial_distances, radial_extreme_percentile)
    )
    radial_extreme_points = points[radial_distances >= robust_outer_radius]

    # 用稳健外径代替单个最大值，避免少量突出 FOV 圆周的尖角整体外推候选带。
    robust_radial_distances = np.minimum(radial_distances, robust_outer_radius)
    radial_threshold = robust_outer_radius - base_radial_band
    base_flags = robust_radial_distances >= radial_threshold

    # 将超出稳健外径的点仅在几何计算中投影回该半径；原始轮廓仍用于显示。
    # 这样保留候选点的角度信息，但尖角额外突出的距离不会拉偏 base。
    safe_distances = np.maximum(radial_distances, 1e-6)
    radial_scales = np.minimum(1.0, robust_outer_radius / safe_distances)
    robust_points = fov_center + offsets * radial_scales[:, None]
    base_runs = _choose_base_runs(
        robust_points,
        _cyclic_true_runs(base_flags),
        fov_center,
        previous_base,
    )
    if base_runs is None:
        return None

    base_candidates = np.concatenate([points[run] for run in base_runs], axis=0)
    base_clusters = [robust_points[run] for run in base_runs]
    base = np.mean(
        np.stack([np.median(cluster, axis=0) for cluster in base_clusters]),
        axis=0,
    )
    centroid = _mask_centroid(mask, points)
    direction = centroid - base
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction /= norm

    projections = (points - base) @ direction
    projection_range = float(np.ptp(projections))
    if projection_range < 5.0:
        return None
    threshold = float(np.percentile(projections, distal_percentile))
    distal_flags = projections >= threshold
    distal_centers = _distal_cluster_centers(points, distal_flags)
    if not distal_centers:
        return None

    # 两个主要远端簇对应张开的两个钳口，取其中点作为功能端。
    tip = np.mean(np.stack(distal_centers), axis=0)
    distal_projection_std = float(np.std(projections[distal_flags]))
    compactness = math.exp(
        -distal_projection_std / max(3.0, 0.03 * projection_range)
    )
    candidate_count = max(1, int(np.count_nonzero(base_flags)))
    base_coherence = min(1.0, len(base_candidates) / candidate_count)
    confidence = float(
        np.clip(
            detection_confidence
            * (0.70 + 0.25 * base_coherence)
            * (0.75 + 0.25 * compactness),
            0.0,
            1.0,
        )
    )
    return TipObservation(
        tip=tip,
        base=base,
        direction=direction,
        confidence=confidence,
        mode="radial_outer",
        distal_centers=distal_centers,
        base_candidates=base_candidates,
        radial_extreme_points=radial_extreme_points,
    )


def _pca_fallback_tip(
    points,
    detection_confidence,
    fov_center,
    previous_tip,
):
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axis = eigenvectors[:, order[0]]
    projections = centered @ axis

    low_threshold = np.percentile(projections, 2.0)
    high_threshold = np.percentile(projections, 98.0)
    low_center = np.median(points[projections <= low_threshold], axis=0)
    high_center = np.median(points[projections >= high_threshold], axis=0)
    candidates = [low_center, high_center]

    if previous_tip is not None:
        tip_index = int(
            np.argmin(
                [np.linalg.norm(candidate - previous_tip) for candidate in candidates]
            )
        )
    else:
        # 没有时序参考时，器械尖端通常比根部更靠近 FOV 中心。
        tip_index = int(
            np.argmin(
                [np.linalg.norm(candidate - fov_center) for candidate in candidates]
            )
        )
    tip = candidates[tip_index]
    base = candidates[1 - tip_index]
    direction = tip - base
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction /= norm

    elongation = float(
        eigenvalues[0] / max(eigenvalues[1], 1e-6)
    )
    elongation_score = float(np.clip((elongation - 1.0) / 8.0, 0.0, 1.0))
    confidence = float(
        np.clip(detection_confidence * (0.35 + 0.25 * elongation_score), 0.0, 1.0)
    )
    return TipObservation(
        tip=tip,
        base=base,
        direction=direction,
        confidence=confidence,
        mode="pca_fallback",
        distal_centers=[tip],
        base_candidates=np.empty((0, 2), dtype=np.float64),
        radial_extreme_points=np.empty((0, 2), dtype=np.float64),
    )


def extract_improved_tip(
    mask,
    detection_confidence,
    fov_center,
    base_radial_band=30.0,
    radial_extreme_percentile=95.0,
    distal_percentile=98.0,
    previous_tip=None,
    previous_base=None,
):
    """从二值掩码中提取改进后的器械尖端。"""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0, :].astype(np.float64)
    if len(points) < 10:
        return None, contour

    observation = _radial_outer_tip(
        mask=mask,
        points=points,
        detection_confidence=detection_confidence,
        fov_center=fov_center,
        base_radial_band=base_radial_band,
        radial_extreme_percentile=radial_extreme_percentile,
        distal_percentile=distal_percentile,
        previous_base=previous_base,
    )
    if observation is None:
        observation = _pca_fallback_tip(
            points=points,
            detection_confidence=detection_confidence,
            fov_center=fov_center,
            previous_tip=previous_tip,
        )
    return observation, contour


def _specular_instance_metrics(
    frame,
    mask,
    value_threshold=250,
    saturation_threshold=8,
):
    """统计候选内部连续的低饱和高亮核心。

    这里只产生实例级判据，不从 mask 中删除高亮像素。真实金属器械上的少量
    离散反光因此会被保留；大块连续过曝组织则会形成较大的连通高光核心。
    """

    mask_bool = np.asarray(mask) > 0
    mask_area = int(np.count_nonzero(mask_bool))
    frame_area = int(mask_bool.size)
    if mask_area == 0 or frame_area == 0:
        return {
            "specular_ratio": 0.0,
            "largest_specular_ratio": 0.0,
            "specular_frame_ratio": 0.0,
            "specular_area": 0,
            "largest_specular_area": 0,
        }

    ys, xs = np.nonzero(mask_bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    mask_roi = mask_bool[y0:y1, x0:x1]
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    specular = (
        mask_roi
        & (value >= int(value_threshold))
        & (saturation <= int(saturation_threshold))
    )
    specular_area = int(np.count_nonzero(specular))
    largest_area = 0
    if specular_area > 0:
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            specular.astype(np.uint8), connectivity=8
        )
        if count > 1:
            largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())

    return {
        "specular_ratio": specular_area / mask_area,
        "largest_specular_ratio": largest_area / mask_area,
        "specular_frame_ratio": largest_area / frame_area,
        "specular_area": specular_area,
        "largest_specular_area": largest_area,
    }


def _select_detection(
    result,
    frame_shape,
    min_area_ratio,
    morph_kernel,
    *,
    frame=None,
    previous_bbox=None,
    track_iou_threshold=0.3,
    specular_filter_enabled=False,
    specular_value_threshold=250,
    specular_saturation_threshold=8,
    specular_mask_ratio=0.70,
    specular_largest_ratio=0.60,
    specular_frame_ratio=0.05,
    return_diagnostics=False,
):
    """选择主器械实例，并保守拒绝由大片连续高光造成的误检。

    ``frame`` 和反光过滤参数均为可选，因此其他复用本函数的独立脚本仍与旧
    调用方式兼容。提供上一帧 bbox 时，候选排序会轻微偏向时序一致的目标。
    """

    h, w = frame_shape[:2]
    diagnostics = {
        "candidate_count": 0,
        "specular_rejected_count": 0,
    }

    def finish(value):
        return (value, diagnostics) if return_diagnostics else value

    if result.masks is None or len(result.masks) == 0:
        return finish(None)
    minimum_area = int(h * w * float(min_area_ratio))
    candidates = []
    boxes = result.boxes
    for index, mask_tensor in enumerate(result.masks.data):
        data = mask_tensor.cpu().numpy() if hasattr(mask_tensor, "cpu") else np.asarray(mask_tensor)
        mask = (data > 0.5).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = _largest_clean_component(mask, morph_kernel)
        if mask is None:
            continue
        area = int(cv2.countNonZero(mask))
        if area < minimum_area:
            continue

        diagnostics["candidate_count"] += 1

        confidence = 0.0
        bbox = None
        if boxes is not None and index < len(boxes):
            confidence = float(boxes.conf[index])
            bbox_data = boxes.xyxy[index]
            bbox = (
                bbox_data.cpu().numpy()
                if hasattr(bbox_data, "cpu")
                else np.asarray(bbox_data)
            )
        if bbox is None:
            ys, xs = np.nonzero(mask)
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
        bbox = np.asarray(bbox, dtype=np.float64)
        metrics = {
            "specular_ratio": 0.0,
            "largest_specular_ratio": 0.0,
            "specular_frame_ratio": 0.0,
            "specular_area": 0,
            "largest_specular_area": 0,
        }
        if frame is not None:
            metrics = _specular_instance_metrics(
                frame,
                mask,
                value_threshold=specular_value_threshold,
                saturation_threshold=specular_saturation_threshold,
            )

        track_iou = (
            0.0
            if previous_bbox is None
            else _box_iou(previous_bbox, bbox)
        )
        track_consistent = bool(
            previous_bbox is not None
            and track_iou >= float(track_iou_threshold)
        )
        specular_rejected = bool(
            specular_filter_enabled
            and frame is not None
            # 已确认轨迹的空间连续候选优先保留，避免器械自身高光导致掉线。
            and not track_consistent
            and metrics["specular_ratio"] >= float(specular_mask_ratio)
            and metrics["largest_specular_ratio"]
            >= float(specular_largest_ratio)
            and metrics["specular_frame_ratio"]
            >= float(specular_frame_ratio)
        )
        if specular_rejected:
            diagnostics["specular_rejected_count"] += 1
            continue

        # 只给历史目标小幅加分；YOLO 置信度仍然是主要排序依据。
        temporal_bonus = 0.20 * track_iou
        candidates.append(
            {
                "mask": mask,
                "bbox": bbox,
                "confidence": confidence,
                "area": area,
                "track_iou": track_iou,
                "track_consistent": track_consistent,
                "selection_score": confidence + temporal_bonus,
                **metrics,
            }
        )
    if not candidates:
        return finish(None)
    # 当前机器人跟随系统以一把主器械为目标。
    selected = max(
        candidates,
        key=lambda item: (
            item["selection_score"],
            item["confidence"],
            item["area"],
        ),
    )
    return finish(selected)


def _circle_from_fov_mask(mask):
    """从 FOV 二值掩码的最大连通轮廓计算最小外接圆。"""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return np.array([cx, cy], dtype=np.float64), float(radius)


def _calibrate_startup_fov(
    capture,
    frame_width,
    frame_height,
    sample_frames,
    skip_frames,
    dark_threshold,
):
    """用视频源开头的多帧亮区自动标定 FOV。

    前 ``skip_frames`` 帧只用于摄像头曝光稳定；后续 ``sample_frames``
    帧使用多帧多数投票生成稳定 FOV 掩码。
    """
    print(
        f"🔄 启动 FOV 标定: 跳过 {skip_frames} 帧，"
        f"采集 {sample_frames} 帧，暗部阈值={dark_threshold}"
    )
    for _ in range(skip_frames):
        ok, _ = capture.read()
        if not ok:
            print("⚠️  FOV 标定在热身阶段无法读取画面")
            return None

    accumulator = np.zeros((frame_height, frame_width), dtype=np.uint32)
    collected = 0
    single_mask = None
    middle_index = sample_frames // 2
    for index in range(sample_frames):
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (frame_height, frame_width):
            frame = cv2.resize(
                frame,
                (frame_width, frame_height),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = gray > dark_threshold
        accumulator += bright.astype(np.uint32)
        if index == middle_index:
            single_mask = bright.astype(np.uint8) * 255
        collected += 1

    if collected < 3:
        print(f"⚠️  FOV 标定只采集到 {collected} 帧，无法可靠计算")
        return None

    vote_threshold = max(1, int(math.ceil(collected * 0.5)))
    average_mask = (accumulator >= vote_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    average_mask = cv2.morphologyEx(average_mask, cv2.MORPH_CLOSE, kernel)
    average_result = _circle_from_fov_mask(average_mask)
    if average_result is None:
        print("⚠️  启动帧中未找到可用的 FOV 亮区轮廓")
        return None

    if single_mask is None:
        single_mask = average_mask.copy()
    else:
        single_mask = cv2.morphologyEx(single_mask, cv2.MORPH_CLOSE, kernel)
    single_result = _circle_from_fov_mask(single_mask) or average_result

    center, radius = average_result
    minimum_radius = 0.2 * min(frame_width, frame_height)
    maximum_radius = 0.8 * max(frame_width, frame_height)
    if not minimum_radius <= radius <= maximum_radius:
        print(
            f"⚠️  启动 FOV 半径 {radius:.1f}px 超出合理范围 "
            f"[{minimum_radius:.1f}, {maximum_radius:.1f}]px"
        )
        return None

    single_center, single_radius = single_result
    print(
        f"✅ 启动 FOV 标定完成: collected={collected}, "
        f"center=({center[0]:.1f}, {center[1]:.1f}), radius={radius:.1f}"
    )
    return {
        "center": center,
        "radius": radius,
        "config": {
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "source": "startup_frames",
            "skip_frames": int(skip_frames),
            "sample_frames": int(collected),
            "dark_threshold": int(dark_threshold),
            "single_frame": {
                "circle_center_x": float(single_center[0]),
                "circle_center_y": float(single_center[1]),
                "circle_radius": float(single_radius),
            },
            "average_frames": {
                "circle_center_x": float(center[0]),
                "circle_center_y": float(center[1]),
                "circle_radius": float(radius),
            },
        },
    }


def _save_startup_fov(config, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    print(f"💾 启动 FOV 配置已保存: {output_path}")


def _load_scaled_fov(config_path, frame_width, frame_height):
    default_center = np.array([frame_width / 2.0, frame_height / 2.0])
    default_radius = min(frame_width, frame_height) / 2.0
    if config_path is None or not config_path.exists():
        print("⚠️  FOV 配置不存在，使用当前画面的默认内切圆")
        return default_center, default_radius
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        source_width = float(data.get("frame_width", frame_width))
        source_height = float(data.get("frame_height", frame_height))
        values = data.get("average_frames", data.get("single_frame", {}))
        scale_x = frame_width / max(source_width, 1.0)
        scale_y = frame_height / max(source_height, 1.0)
        center = np.array(
            [
                float(values["circle_center_x"]) * scale_x,
                float(values["circle_center_y"]) * scale_y,
            ],
            dtype=np.float64,
        )
        radius = float(values["circle_radius"]) * 0.5 * (scale_x + scale_y)
        print(
            f"✅ FOV: center=({center[0]:.1f}, {center[1]:.1f}), "
            f"radius={radius:.1f}"
        )
        return center, radius
    except Exception as exc:
        print(f"⚠️  FOV 配置读取失败，使用默认内切圆: {exc}")
        return default_center, default_radius


def _draw_fov_boundary(image, fov_center, fov_radius, color=(255, 180, 0)):
    """绘制「圆盘 ∩ 图像矩形」的完整边界。"""
    if fov_radius <= 0:
        return
    h, w = image.shape[:2]
    cx, cy = map(float, fov_center)
    radius = float(fov_radius)
    cv2.circle(
        image,
        (int(round(cx)), int(round(cy))),
        int(round(radius)),
        color,
        1,
        cv2.LINE_AA,
    )

    # 理论圆超出画面时，画出真实可见 FOV 的截断弦线。
    for y in (0, h - 1):
        dy = float(y) - cy
        if abs(dy) < radius and (
            (y == 0 and cy - radius < 0)
            or (y == h - 1 and cy + radius > h - 1)
        ):
            dx = math.sqrt(max(0.0, radius * radius - dy * dy))
            x1 = int(np.clip(round(cx - dx), 0, w - 1))
            x2 = int(np.clip(round(cx + dx), 0, w - 1))
            cv2.line(image, (x1, y), (x2, y), color, 1, cv2.LINE_AA)

    for x in (0, w - 1):
        dx = float(x) - cx
        if abs(dx) < radius and (
            (x == 0 and cx - radius < 0)
            or (x == w - 1 and cx + radius > w - 1)
        ):
            dy = math.sqrt(max(0.0, radius * radius - dx * dx))
            y1 = int(np.clip(round(cy - dy), 0, h - 1))
            y2 = int(np.clip(round(cy + dy), 0, h - 1))
            cv2.line(image, (x, y1), (x, y2), color, 1, cv2.LINE_AA)


def _draw_mask_centroid(image, mask):
    """计算并绘制二值掩码的面积质心。"""
    moments = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 1e-6:
        return None

    cx = int(round(moments["m10"] / moments["m00"]))
    cy = int(round(moments["m01"] / moments["m00"]))
    centroid = (cx, cy)

    # 黑色外轮廓 + 白色十字，在高亮反光和红色组织上都保持可见。
    cv2.drawMarker(
        image, centroid, (0, 0, 0), cv2.MARKER_CROSS, 24, 5, cv2.LINE_AA
    )
    cv2.drawMarker(
        image, centroid, (255, 255, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA
    )

    text = f"MASK CENTROID ({cx}, {cy})"
    image_h, image_w = image.shape[:2]
    text_x = cx + 14 if cx < image_w - 330 else max(5, cx - 325)
    text_y = int(np.clip(cy - 12, 24, image_h - 8))
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return centroid


def _draw_tracking_thresholds(
    image,
    target_pixel,
    start_threshold,
    stop_threshold,
):
    """在原始视频上绘制与 MATLAB 配置一致的跟踪启停参考圈。"""
    h, w = image.shape[:2]
    target = np.asarray(target_pixel, dtype=np.float64)
    center = (
        int(np.clip(round(target[0]), 0, w - 1)),
        int(np.clip(round(target[1]), 0, h - 1)),
    )
    start_radius = max(0, int(round(start_threshold)))
    stop_radius = max(0, int(round(stop_threshold)))

    if start_radius > 0:
        cv2.circle(image, center, start_radius, (0, 165, 255), 2, cv2.LINE_AA)
    if stop_radius > 0:
        cv2.circle(image, center, stop_radius, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(
        image, center, (255, 255, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA
    )

    label = (
        f"TRACK TARGET ({center[0]}, {center[1]})  "
        f"STOP={stop_radius}px  START={start_radius}px"
    )
    text_y = int(np.clip(106, 24, max(24, h - 12)))
    cv2.putText(
        image,
        label,
        (16, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (16, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_scope_motion_arrow(image, fov_center, fov_radius, scope_motion):
    """从FOV中心绘制MATLAB回传的内窥镜运动方向短箭头。"""
    if scope_motion is None:
        return

    motion = np.asarray(
        [scope_motion.get("du", 0.0), scope_motion.get("dv", 0.0)],
        dtype=np.float64,
    )
    speed = float(np.linalg.norm(motion))
    if not np.isfinite(speed) or speed < 2.0:
        return

    h, w = image.shape[:2]
    center = np.asarray(fov_center, dtype=np.float64)
    arrow_length = float(np.clip(0.07 * float(fov_radius), 28.0, 48.0))
    tip = center + motion / speed * arrow_length
    start_point = (
        int(np.clip(round(center[0]), 0, w - 1)),
        int(np.clip(round(center[1]), 0, h - 1)),
    )
    end_point = (
        int(np.clip(round(tip[0]), 0, w - 1)),
        int(np.clip(round(tip[1]), 0, h - 1)),
    )

    # 黑色轮廓保证亮背景下仍清晰，青色与器械方向的品红箭头区分。
    cv2.arrowedLine(
        image, start_point, end_point, (0, 0, 0), 6, cv2.LINE_AA, tipLength=0.28
    )
    cv2.arrowedLine(
        image,
        start_point,
        end_point,
        (255, 255, 0),
        3,
        cv2.LINE_AA,
        tipLength=0.28,
    )


def _draw_result(
    frame,
    detection,
    contour,
    observation,
    tracked,
    fov_center,
    fov_radius,
    tracking_target,
    start_threshold,
    stop_threshold,
    scope_motion=None,
    selection_diagnostics=None,
):
    annotated = frame.copy()
    if detection is not None:
        mask_region = detection["mask"] > 0
        annotated[mask_region] = (
            annotated[mask_region].astype(np.float32) * (1.0 - MASK_ALPHA)
            + MASK_COLOR * MASK_ALPHA
        ).astype(np.uint8)
        if contour is not None:
            cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        _draw_mask_centroid(annotated, detection["mask"])

    _draw_fov_boundary(annotated, fov_center, fov_radius)
    _draw_tracking_thresholds(
        annotated,
        target_pixel=tracking_target,
        start_threshold=start_threshold,
        stop_threshold=stop_threshold,
    )
    _draw_scope_motion_arrow(annotated, fov_center, fov_radius, scope_motion)

    if observation is not None:
        if len(observation.base_candidates) > 0:
            for candidate in observation.base_candidates:
                point = tuple(np.rint(candidate).astype(int))
                cv2.circle(annotated, point, 2, (0, 165, 255), -1)
        if len(observation.radial_extreme_points) > 0:
            for extreme_point in observation.radial_extreme_points:
                point = tuple(np.rint(extreme_point).astype(int))
                cv2.circle(annotated, point, 4, RADIAL_EXTREME_COLOR, -1)
        base = tuple(np.rint(observation.base).astype(int))
        raw_tip = tuple(np.rint(observation.tip).astype(int))
        cv2.circle(annotated, base, 7, (0, 165, 255), -1)
        cv2.circle(annotated, raw_tip, 5, (0, 255, 255), -1)
        for center in observation.distal_centers:
            cv2.circle(
                annotated,
                tuple(np.rint(center).astype(int)),
                4,
                (255, 255, 0),
                1,
            )

    if tracked is not None and tracked.valid:
        tip = tuple(np.rint(tracked.tip).astype(int))
        if tracked.base is not None:
            base = tuple(np.rint(tracked.base).astype(int))
            cv2.arrowedLine(
                annotated, base, tip, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.05
            )
        cv2.circle(annotated, tip, 11, (255, 0, 255), -1)
        cv2.circle(annotated, tip, 11, (255, 255, 255), 2)
        cv2.putText(
            annotated,
            f"TIP ({tip[0]}, {tip[1]})",
            (tip[0] + 14, max(24, tip[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 0, 255),
            2,
        )
        status = f"mode={tracked.mode}  tip_conf={tracked.confidence:.2f}"
    else:
        status = "tip unavailable"

    cv2.putText(
        annotated,
        status,
        (16, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    if detection is not None:
        specular_ratio = float(detection.get("largest_specular_ratio", 0.0))
        cv2.putText(
            annotated,
            (
                f"seg_conf={detection['confidence']:.2f}  "
                f"spec_core={specular_ratio:.3f}"
            ),
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 0),
            2,
        )
    elif (
        selection_diagnostics is not None
        and selection_diagnostics.get("specular_rejected_count", 0) > 0
    ):
        cv2.putText(
            annotated,
            (
                "seg rejected: specular="
                f"{selection_diagnostics['specular_rejected_count']}/"
                f"{selection_diagnostics['candidate_count']}"
            ),
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 165, 255),
            2,
        )
    return annotated


def _resolve_path(project_root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _build_writer(output_path, fps, width, height):
    suffix = output_path.suffix.lower()
    if suffix not in {".mp4", ".avi"}:
        raise ValueError("输出视频仅支持 .mp4 或 .avi")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = "mp4v" if suffix == ".mp4" else "XVID"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"无法创建输出视频: {output_path}")
    return writer


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11-seg 实时分割 + 改进器械尖端几何检测"
    )
    parser.add_argument("--video", default=None, help="输入视频；不指定时使用摄像头")
    parser.add_argument("--camera", type=int, default=0, help="摄像头/采集卡编号")
    parser.add_argument(
        "--model",
        default=(
            "larger_surgical_video_dataset_training_20260712/training_results/"
            "yolo11/yolo11l_updated_20260712_223026/weights/best.pt"
        ),
        help="YOLO-seg 权重路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="带 overlay 的输出 .mp4/.avi 路径",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help="不带 overlay 的原始视频 .mp4/.avi 路径",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--min-area", type=float, default=0.001)
    parser.add_argument("--morph-kernel", type=int, default=5)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--track-iou", type=float, default=0.3)
    parser.add_argument(
        "--specular-filter",
        action="store_true",
        help="开启大面积连续高反光误检实例过滤（默认关闭）",
    )
    parser.add_argument(
        "--specular-value-threshold",
        type=int,
        default=230,
        help="HSV V 强过曝核心阈值 (默认: 250)",
    )
    parser.add_argument(
        "--specular-saturation-threshold",
        type=int,
        default=8,
        help="HSV S 强过曝核心上限 (默认: 8)",
    )
    parser.add_argument(
        "--specular-mask-ratio",
        type=float,
        default=0.70,
        help="高光核心总面积/mask 面积拒绝阈值 (默认: 0.70)",
    )
    parser.add_argument(
        "--specular-largest-ratio",
        type=float,
        default=0.60,
        help="最大连续高光核心/mask 面积拒绝阈值 (默认: 0.60)",
    )
    parser.add_argument(
        "--specular-frame-ratio",
        type=float,
        default=0.05,
        help="最大连续高光核心/整帧面积拒绝阈值 (默认: 0.05)",
    )
    parser.add_argument(
        "--base-radial-band",
        "--boundary-thickness",
        dest="base_radial_band",
        type=float,
        default=30.0,
        help=(
            "最外径向点以内的根部候选带宽，像素 (默认: 30；"
            "--boundary-thickness 作为兼容别名)"
        ),
    )
    parser.add_argument(
        "--radial-extreme-percentile",
        type=float,
        default=95.0,
        help="蓝色显示的最外侧径向轮廓百分位 (默认: 95，即最高 5%%)",
    )
    parser.add_argument("--distal-percentile", type=float, default=98.0)
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.2)
    parser.add_argument("--one-euro-beta", type=float, default=0.015)
    parser.add_argument("--max-tip-jump", type=float, default=120.0)
    parser.add_argument("--hold-frames", type=int, default=1)
    parser.add_argument(
        "--reacquire-frames",
        type=int,
        default=3,
        help="大跳变后新尖端连续多少帧一致才重新捕获 (默认: 3)",
    )
    parser.add_argument(
        "--reacquire-radius",
        type=float,
        default=80.0,
        help="重捕获候选的帧间一致性半径，像素 (默认: 80)",
    )
    parser.add_argument(
        "--startup-fov-frames",
        type=int,
        default=100,
        help="启动时用于自动标定 FOV 的帧数；0 表示关闭 (默认: 60)",
    )
    parser.add_argument(
        "--startup-fov-skip",
        type=int,
        default=20,
        help="FOV 标定前丢弃的摄像头热身帧数 (默认: 10)",
    )
    parser.add_argument(
        "--fov-dark-threshold",
        type=int,
        default=30,
        help="启动 FOV 标定中区分有效视野与黑边的灰度阈值 (默认: 20)",
    )
    parser.add_argument(
        "--startup-fov-output",
        default=None,
        help="可选：将本次启动标定结果另存为 JSON",
    )
    parser.add_argument(
        "--fov-config",
        default="results/fov_config.json",
        help="关闭启动标定或标定失败时使用的 FOV 配置",
    )
    parser.add_argument(
        "--target-pixel",
        type=float,
        nargs=2,
        default=(1001.0, 473.0),
        metavar=("U", "V"),
        help="MATLAB期望像素点，用作启停圈圆心 (默认: 1001 473)",
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=250.0,
        help="开始跟踪参考圈半径，像素 (默认: 300)",
    )
    parser.add_argument(
        "--stop-threshold",
        type=float,
        default=50.0,
        help="停止跟踪参考圈半径，像素 (默认: 60)",
    )
    parser.add_argument(
        "--zmq-port", type=int, default=None, help="可选 ZMQ PUB 端口，例如 5556"
    )
    parser.add_argument(
        "--matlab-host",
        default="127.0.0.1",
        help="MATLAB tcpclient 连接地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--matlab-port",
        type=int,
        default=5555,
        help="MATLAB 普通 TCP JSON 端口 (默认: 5555)",
    )
    parser.add_argument(
        "--no-matlab-tcp",
        action="store_true",
        help="关闭默认的 MATLAB TCP JSON 输出",
    )
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if args.base_radial_band <= 0.0:
        parser.error("--base-radial-band 必须 > 0")
    if not 0.0 < args.radial_extreme_percentile < 100.0:
        parser.error("--radial-extreme-percentile 必须位于 (0, 100)")
    if not 0.0 < args.distal_percentile < 100.0:
        parser.error("--distal-percentile 必须位于 (0, 100)")
    if args.confirm_frames < 1:
        parser.error("--confirm-frames 必须 >= 1")
    if not 0.0 <= args.track_iou <= 1.0:
        parser.error("--track-iou 必须位于 [0, 1]")
    if not 0 <= args.specular_value_threshold <= 255:
        parser.error("--specular-value-threshold 必须位于 [0, 255]")
    if not 0 <= args.specular_saturation_threshold <= 255:
        parser.error("--specular-saturation-threshold 必须位于 [0, 255]")
    for name in (
        "specular_mask_ratio",
        "specular_largest_ratio",
        "specular_frame_ratio",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} 必须位于 [0, 1]")
    if args.reacquire_frames < 1:
        parser.error("--reacquire-frames 必须 >= 1")
    if args.reacquire_radius <= 0:
        parser.error("--reacquire-radius 必须 > 0")
    if args.startup_fov_frames < 0:
        parser.error("--startup-fov-frames 必须 >= 0")
    if args.startup_fov_skip < 0:
        parser.error("--startup-fov-skip 必须 >= 0")
    if not 0 <= args.fov_dark_threshold <= 255:
        parser.error("--fov-dark-threshold 必须位于 [0, 255]")
    if args.stop_threshold < 0:
        parser.error("--stop-threshold 必须 >= 0")
    if args.start_threshold <= args.stop_threshold:
        parser.error("--start-threshold 必须大于 --stop-threshold")
    if not args.no_matlab_tcp and not 1 <= args.matlab_port <= 65535:
        parser.error("--matlab-port 必须位于 [1, 65535]")
    return args


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    model_path = _resolve_path(project_root, args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型: {model_path}")
    is_video = bool(args.video)
    if is_video:
        source_path = _resolve_path(project_root, args.video)
        capture = cv2.VideoCapture(str(source_path))
    else:
        source_path = None
        capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入: {source_path or args.camera}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    camera_fourcc = None
    if not is_video:
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        camera_fourcc = "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
        ).rstrip("\x00")
        if camera_fourcc != "MJPG" or fps < 29.0:
            print(
                f"⚠️  摄像头未完全达到请求格式: "
                f"fourcc={camera_fourcc or '未知'}, fps={fps:.2f}"
            )
    nominal_dt = 1.0 / max(fps, 1.0)

    fov_path = _resolve_path(project_root, args.fov_config) if args.fov_config else None
    startup_fov = None
    if args.startup_fov_frames > 0:
        startup_fov = _calibrate_startup_fov(
            capture=capture,
            frame_width=width,
            frame_height=height,
            sample_frames=args.startup_fov_frames,
            skip_frames=args.startup_fov_skip,
            dark_threshold=args.fov_dark_threshold,
        )

    if is_video and args.startup_fov_frames > 0:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            print("⚠️  离线视频标定后无法回到第 0 帧")

    if startup_fov is not None:
        fov_center = startup_fov["center"]
        fov_radius = startup_fov["radius"]
        if args.startup_fov_output:
            startup_fov_path = _resolve_path(project_root, args.startup_fov_output)
            _save_startup_fov(startup_fov["config"], startup_fov_path)
    else:
        if args.startup_fov_frames > 0:
            print("⚠️  启动 FOV 标定失败，回退到 --fov-config")
        fov_center, fov_radius = _load_scaled_fov(fov_path, width, height)

    output_path = None
    if args.output:
        output_path = _resolve_path(project_root, args.output).resolve()
    elif is_video:
        output_path = (
            project_root
            / "segnext_instrument_seg/outputs/sugical_output"
            / f"{source_path.stem}_yolo11_improved_tip.mp4"
        ).resolve()
    if is_video and output_path is not None and output_path == source_path.resolve():
        raise ValueError("输出路径不能与输入视频相同")

    raw_output_path = (
        _resolve_path(project_root, args.raw_output).resolve()
        if args.raw_output
        else None
    )
    if is_video and raw_output_path is not None and raw_output_path == source_path.resolve():
        raise ValueError("原始视频输出路径不能与输入视频相同")
    if (
        output_path is not None
        and raw_output_path is not None
        and output_path == raw_output_path
    ):
        raise ValueError("带 overlay 和不带 overlay 的输出路径不能相同")

    overlay_writer = (
        _build_writer(output_path, fps, width, height)
        if output_path is not None
        else None
    )
    raw_writer = (
        _build_writer(raw_output_path, fps, width, height)
        if raw_output_path is not None
        else None
    )
    publisher = TipPublisher(args.zmq_port)
    matlab_publisher = MatlabTcpPublisher(
        host=args.matlab_host,
        port=None if args.no_matlab_tcp else args.matlab_port,
    )
    model = YOLO(str(model_path))
    tracker = TemporalTipTracker(
        min_cutoff=args.one_euro_min_cutoff,
        beta=args.one_euro_beta,
        max_jump=args.max_tip_jump,
        hold_frames=args.hold_frames,
        reacquire_frames=args.reacquire_frames,
        reacquire_radius=args.reacquire_radius,
    )
    detection_gate = DetectionGate(args.confirm_frames, args.track_iou)

    print(f"模型: {model_path}")
    print(f"输入: {source_path if is_video else f'/dev/video{args.camera}'}")
    print(f"分辨率: {width}x{height} @ {fps:.2f} FPS")
    if camera_fourcc is not None:
        print(f"采集格式: {camera_fourcc or '未知'}")
    print(f"Overlay 输出: {output_path or '不保存'}")
    print(f"原始视频输出: {raw_output_path or '不保存'}")
    print(
        "高反光过滤: "
        + (
            (
                f"开启 (V>={args.specular_value_threshold}, "
                f"S<={args.specular_saturation_threshold}, "
                f"mask>={args.specular_mask_ratio:.3f}, "
                f"largest>={args.specular_largest_ratio:.3f}, "
                f"frame>={args.specular_frame_ratio:.3f})"
            )
            if args.specular_filter
            else "关闭（使用 --specular-filter 开启）"
        )
    )
    print("按 q 退出")

    frame_index = 0
    specular_candidates_total = 0
    specular_rejected_total = 0
    previous_tracker_time = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            result = model(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            inference_end = time.perf_counter()

            detection, selection_diagnostics = _select_detection(
                result,
                frame.shape,
                args.min_area,
                args.morph_kernel,
                frame=frame,
                previous_bbox=detection_gate.last_bbox,
                track_iou_threshold=args.track_iou,
                specular_filter_enabled=args.specular_filter,
                specular_value_threshold=args.specular_value_threshold,
                specular_saturation_threshold=(
                    args.specular_saturation_threshold
                ),
                specular_mask_ratio=args.specular_mask_ratio,
                specular_largest_ratio=args.specular_largest_ratio,
                specular_frame_ratio=args.specular_frame_ratio,
                return_diagnostics=True,
            )
            specular_candidates_total += selection_diagnostics[
                "candidate_count"
            ]
            specular_rejected_total += selection_diagnostics[
                "specular_rejected_count"
            ]
            observation = None
            contour = None
            if detection is None:
                detection_gate.reset()
            else:
                confirmed, switched = detection_gate.update(detection["bbox"])
                if switched:
                    tracker.reset()
                if confirmed:
                    observation, contour = extract_improved_tip(
                        mask=detection["mask"],
                        detection_confidence=detection["confidence"],
                        fov_center=fov_center,
                        base_radial_band=args.base_radial_band,
                        radial_extreme_percentile=args.radial_extreme_percentile,
                        distal_percentile=args.distal_percentile,
                        previous_tip=tracker.reference_tip,
                        previous_base=tracker.reference_base,
                    )

            tracker_time = time.perf_counter()
            tracker_dt = (
                nominal_dt
                if previous_tracker_time is None
                else max(tracker_time - previous_tracker_time, 1e-6)
            )
            previous_tracker_time = tracker_time
            tracked = tracker.update(observation, tracker_dt)
            scope_motion = matlab_publisher.receive_scope_motion()
            annotated = _draw_result(
                frame,
                detection,
                contour,
                observation,
                tracked,
                fov_center,
                fov_radius,
                tracking_target=args.target_pixel,
                start_threshold=args.start_threshold,
                stop_threshold=args.stop_threshold,
                scope_motion=scope_motion,
                selection_diagnostics=selection_diagnostics,
            )
            if tracked is not None:
                publisher.send(tracked)
                matlab_publisher.send(tracked, frame_index=frame_index)
            if overlay_writer is not None:
                overlay_writer.write(annotated)
            if raw_writer is not None:
                raw_writer.write(frame)

            frame_index += 1
            end = time.perf_counter()
            if frame_index % 30 == 0:
                print(
                    f"[PROFILE] frame={frame_index} "
                    f"YOLO={(inference_end-start)*1000:.1f}ms "
                    f"post={(end-inference_end)*1000:.1f}ms "
                    f"total={(end-start)*1000:.1f}ms"
                )

            if not args.no_display:
                cv2.imshow("Improved instrument tip detection", annotated)
                delay = max(1, int(round(1000.0 / fps))) if is_video else 1
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        capture.release()
        if overlay_writer is not None:
            overlay_writer.release()
        if raw_writer is not None:
            raw_writer.release()
        publisher.close()
        matlab_publisher.close()
        cv2.destroyAllWindows()

    print(f"✅ Overlay 视频: {output_path or '未保存'}")
    print(f"✅ 原始视频: {raw_output_path or '未保存'}")
    if args.specular_filter:
        print(
            "✅ 高反光过滤统计: "
            f"rejected={specular_rejected_total}/"
            f"candidates={specular_candidates_total}"
        )


if __name__ == "__main__":
    main()
