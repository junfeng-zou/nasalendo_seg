#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from distance_state_classifier.src.config import get_nested, load_config, resolve_path
from distance_state_classifier.src.predictor import DistanceStatePredictor, Prediction
from distance_state_classifier.src.transforms import apply_crop


STATE_COLORS = {
    "TooFar": (255, 180, 60),
    "Good": (80, 220, 90),
    "TooClose": (60, 80, 255),
    "Invalid": (170, 170, 170),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time distance-state inference on a camera stream.")
    parser.add_argument("--config", default="distance_state_classifier/configs/distance_state_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera", default="0", help="Camera index, video device path, RTSP URL, or video file.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG", help="Requested camera fourcc, e.g. MJPG or YUYV.")
    parser.add_argument("--apply-crop", action="store_true", help="Apply raw-frame crop before inference.")
    parser.add_argument("--display-crop", action="store_true", help="Show the cropped effective field of view.")
    parser.add_argument("--infer-every", type=int, default=1, help="Run model every N frames; reuse last result between runs.")
    parser.add_argument("--preview-scale", type=float, default=0.5)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-video", "--record-video", dest="save_video", default=None, help="Annotated video output path used when recording is started.")
    parser.add_argument("--record-from-start", action="store_true", help="Record the annotated stream from the first frame.")
    parser.add_argument("--save-jsonl", default=None, help="Optional per-inference jsonl output path.")
    parser.add_argument("--yolo-model", default=None, help="YOLO segmentation checkpoint for instrument-mask gating.")
    parser.add_argument("--mask-conf", type=float, default=None)
    parser.add_argument("--mask-iou", type=float, default=None)
    parser.add_argument("--mask-imgsz", type=int, default=None)
    parser.add_argument("--mask-device", default=None, help="YOLO device, e.g. 0, cuda:0, or cpu.")
    parser.add_argument("--mask-class-ids", default=None, help="Comma-separated YOLO class ids to keep.")
    parser.add_argument("--display-mask", action="store_true", help="Overlay the YOLO instrument mask on the preview/recording.")
    parser.add_argument("--mask-alpha", type=float, default=0.35)
    parser.add_argument("--no-instrument-gate", action="store_true", help="Disable YOLO no-instrument gating.")
    parser.add_argument("--depth-window", type=int, default=8, help="Rolling window size for displayed depth_raw.")
    parser.add_argument("--print-every", type=int, default=30)
    return parser.parse_args()


def parse_camera_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(parse_camera_source(str(args.camera)))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera/video source: {args.camera}")

    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc[:4]))
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def make_writer(path: str | Path | None, fps: float, frame_size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path] | None:
    if not path:
        return None
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    candidates: list[tuple[Path, str]] = []
    if suffix in {".mp4", ".m4v"}:
        candidates.extend((output_path, fourcc) for fourcc in ("mp4v", "avc1", "H264"))
        fallback_path = output_path.with_suffix(".avi")
        candidates.extend((fallback_path, fourcc) for fourcc in ("XVID", "MJPG"))
    else:
        candidates.extend((output_path, fourcc) for fourcc in ("XVID", "MJPG", "mp4v"))

    tried = []
    for candidate_path, fourcc_name in candidates:
        writer = cv2.VideoWriter(
            str(candidate_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            max(fps, 1.0),
            frame_size,
        )
        if writer.isOpened():
            if candidate_path != output_path:
                print(f"[camera] video writer fallback: {output_path} -> {candidate_path} ({fourcc_name})")
            else:
                print(f"[camera] video writer opened: {candidate_path} ({fourcc_name})")
            return writer, candidate_path
        writer.release()
        tried.append(f"{candidate_path} ({fourcc_name})")

    raise SystemExit("Cannot open video writer. Tried: " + ", ".join(tried))


def default_record_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolve_path(f"distance_state_classifier/runs/camera_record_{stamp}.mp4")


def parse_class_ids(value: str | None, config_value) -> set[int] | None:
    if value is not None:
        value = value.strip()
        if not value:
            return None
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    if config_value:
        return {int(item) for item in config_value}
    return None


def union_result_masks(result, image_shape: tuple[int, int], class_ids: set[int] | None = None) -> np.ndarray:
    h, w = image_shape
    union = np.zeros((h, w), dtype=np.uint8)
    if getattr(result, "masks", None) is None or result.masks is None:
        return union
    masks = result.masks.data
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)

    keep = range(len(masks))
    if class_ids is not None and getattr(result, "boxes", None) is not None and result.boxes is not None:
        cls = result.boxes.cls
        if hasattr(cls, "detach"):
            cls = cls.detach().cpu().numpy()
        keep = [idx for idx, value in enumerate(cls.tolist()) if int(value) in class_ids]

    for idx in keep:
        resized = cv2.resize(masks[idx], (w, h), interpolation=cv2.INTER_NEAREST)
        union[resized > 0.5] = 255
    return union


class YoloMaskGenerator:
    def __init__(self, args: argparse.Namespace, config: dict) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit("Instrument-mask gating requires ultralytics in the active environment.") from exc

        self.model_path = resolve_path(args.yolo_model or get_nested(config, "mask.yolo_model", "runs/segment/runs/yolo11s_seg_formal/weights/best.pt"))
        self.conf = float(args.mask_conf if args.mask_conf is not None else get_nested(config, "mask.conf", 0.25))
        self.iou = float(args.mask_iou if args.mask_iou is not None else get_nested(config, "mask.iou", 0.7))
        self.imgsz = int(args.mask_imgsz if args.mask_imgsz is not None else get_nested(config, "mask.imgsz", 1024))
        self.device = str(args.mask_device if args.mask_device is not None else get_nested(config, "mask.device", "0"))
        self.class_ids = parse_class_ids(args.mask_class_ids, get_nested(config, "mask.class_ids", None))
        self.model = YOLO(str(self.model_path))

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return union_result_masks(results[0], frame_bgr.shape[:2], class_ids=self.class_ids)


def format_probabilities(result: Prediction | None) -> str:
    if result is None:
        return "waiting..."
    if result.probabilities and all(np.isnan(float(prob)) for prob in result.probabilities.values()):
        return "probabilities=NaN"
    parts = [f"{label}:{prob:.2f}" for label, prob in result.probabilities.items()]
    return "  ".join(parts)


def depth_raw_value(result: Prediction | None) -> float | None:
    if result is None:
        return None
    return float(result.probabilities.get("TooFar", 0.0) - result.probabilities.get("TooClose", 0.0))


def has_instrument_mask(mask: np.ndarray | None) -> bool:
    return mask is not None and mask.size > 0 and np.count_nonzero(mask) > 0


def make_invalid_prediction(classes: list[str]) -> Prediction:
    nan = float("nan")
    return Prediction(
        raw_label="Invalid",
        raw_confidence=0.0,
        state="Invalid",
        smoothed_state="Invalid",
        probabilities={label: nan for label in classes},
    )


def average_depth_raw(history: deque[float]) -> float | None:
    if not history:
        return None
    return float(sum(history) / len(history))


def format_depth_raw(value: float | None, window: int) -> str:
    label = f"depth_raw{window}" if window > 1 else "depth_raw"
    if value is None:
        return f"{label}=--"
    if np.isnan(value):
        return f"{label}=NaN"
    return f"{label}={value:+.3f}"


def blend_mask(frame: np.ndarray, mask: np.ndarray | None, alpha: float = 0.35) -> None:
    if mask is None:
        return
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    active = mask > 0
    if not np.any(active):
        return
    alpha = max(0.0, min(1.0, float(alpha)))
    color = np.zeros_like(frame)
    color[:, :, 1] = 255
    frame[active] = cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0)[active]


def draw_overlay(
    frame,
    result: Prediction | None,
    measured_fps: float,
    latency_ms: float | None,
    frame_index: int,
    recording: bool,
    mask_enabled: bool,
    depth_raw: float | None,
    depth_window: int,
) -> None:
    if result is None:
        state = "Waiting"
        raw_text = ""
        prob_text = "warming up"
    else:
        state = result.smoothed_state
        raw_text = f"raw={result.raw_label} conf={result.raw_confidence:.3f} state={result.state}"
        prob_text = format_probabilities(result)

    color = STATE_COLORS.get(state, (255, 255, 255))
    latency_text = "--" if latency_ms is None else f"{latency_ms:.1f} ms"
    infer_fps_text = "--" if latency_ms is None or latency_ms <= 0 else f"{1000.0 / latency_ms:.1f}"
    lines = [
        f"Distance: {state}",
        raw_text,
        f"Display FPS: {measured_fps:.1f}",
        f"Infer FPS: {infer_fps_text}  latency={latency_text}  frame={frame_index}",
        format_depth_raw(depth_raw, depth_window),
        prob_text,
        ("Mask: ON" if mask_enabled else "Mask: OFF") + ("  REC: ON  q=stop  Esc=quit" if recording else "  REC: OFF  q=start  Esc=quit"),
    ]
    lines = [line for line in lines if line]

    x, y = 18, 34
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    line_h = 31
    box_w = min(frame.shape[1] - 20, 760)
    box_h = 18 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + idx * line_h), font, font_scale, color if idx == 0 else (245, 245, 245), thickness, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (frame.shape[1] - 34, 34), 10, (0, 0, 255), -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    predictor = DistanceStatePredictor(args.checkpoint, config, device=args.device)
    infer_every = max(1, int(args.infer_every))
    depth_window = max(1, int(args.depth_window))
    instrument_gate_enabled = not args.no_instrument_gate
    mask_generator = YoloMaskGenerator(args, config) if instrument_gate_enabled else None

    cap = open_capture(args)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps or 30.0)
    print(f"[camera] source={args.camera} size={source_width}x{source_height} fps={source_fps:.2f}")
    print(f"[camera] checkpoint={resolve_path(args.checkpoint)}")
    if mask_generator is not None:
        print(f"[camera] instrument gate enabled, yolo={mask_generator.model_path}")
    record_path = resolve_path(args.save_video) if args.save_video else default_record_path()
    if args.save_video and args.no_display and not args.record_from_start:
        print("[camera] --save-video is set, but recording is started by q. Remove --no-display or add --record-from-start.")
    if not args.no_display:
        print("[camera] press q to start/stop recording, Esc to stop")
        print(f"[camera] recording output: {record_path}")

    jsonl_file = None
    if args.save_jsonl:
        jsonl_path = resolve_path(args.save_jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8")

    writer = None
    active_record_path = record_path
    recording = bool(args.record_from_start)
    if recording:
        print(f"[camera] recording from start: {record_path}")
    frame_index = 0
    last_result: Prediction | None = None
    last_latency_ms: float | None = None
    last_mask: np.ndarray | None = None
    depth_raw_history: deque[float] = deque(maxlen=depth_window)
    last_depth_raw_window: float | None = None
    started = time.time()
    last_report = started
    recent_times: list[float] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera] failed to read frame, stopping")
                break

            frame_index += 1
            now = time.time()
            recent_times.append(now)
            recent_times = recent_times[-30:]
            if len(recent_times) >= 2:
                measured_fps = (len(recent_times) - 1) / max(recent_times[-1] - recent_times[0], 1e-6)
            else:
                measured_fps = 0.0

            should_infer = (frame_index - 1) % infer_every == 0
            if should_infer:
                t0 = time.time()
                last_mask = mask_generator.predict(frame) if mask_generator is not None else None
                no_instrument = instrument_gate_enabled and mask_generator is not None and not has_instrument_mask(last_mask)
                if no_instrument:
                    predictor.smoother.history.clear()
                    predictor.smoother.current_state = "Invalid"
                    depth_raw_history.clear()
                    last_result = make_invalid_prediction(list(predictor.classes))
                else:
                    last_result = predictor.predict_array(frame, apply_raw_crop=args.apply_crop)
                last_latency_ms = (time.time() - t0) * 1000.0
                current_depth_raw = float("nan") if no_instrument else depth_raw_value(last_result)
                if no_instrument:
                    last_depth_raw_window = float("nan")
                else:
                    if current_depth_raw is not None:
                        depth_raw_history.append(current_depth_raw)
                    last_depth_raw_window = average_depth_raw(depth_raw_history)
                if jsonl_file is not None:
                    jsonl_file.write(
                        json.dumps(
                            {
                                "frame_index": frame_index,
                                "time_sec": time.time() - started,
                                "raw_label": last_result.raw_label,
                                "raw_confidence": last_result.raw_confidence,
                                "state": last_result.state,
                                "smoothed_state": last_result.smoothed_state,
                                "probabilities": last_result.probabilities,
                                "depth_raw": current_depth_raw,
                                "depth_raw_window": last_depth_raw_window,
                                "depth_window": depth_window,
                                "latency_ms": last_latency_ms,
                                "no_instrument": no_instrument,
                                "mask_area_ratio": float(np.count_nonzero(last_mask) / last_mask.size) if last_mask is not None and last_mask.size > 0 else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    jsonl_file.flush()

            base_view = frame
            if args.display_crop:
                base_view = apply_crop(base_view, predictor.crop)
            mask_view = None
            if last_mask is not None:
                mask_view = apply_crop(last_mask, predictor.crop) if args.display_crop else last_mask
            base_view = base_view.copy()
            if args.display_mask:
                blend_mask(base_view, mask_view, alpha=args.mask_alpha)
            view = base_view.copy()
            draw_overlay(view, last_result, measured_fps, last_latency_ms, frame_index, recording, mask_generator is not None, last_depth_raw_window, depth_window)

            if args.preview_scale > 0 and args.preview_scale != 1.0:
                preview = cv2.resize(
                    view,
                    None,
                    fx=args.preview_scale,
                    fy=args.preview_scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                preview = view

            if not args.no_display:
                cv2.imshow("Distance State Camera Inference", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    recording = not recording
                    state_text = "started" if recording else "stopped"
                    print(f"[camera] recording {state_text}: {record_path}")
                    view = base_view.copy()
                    draw_overlay(view, last_result, measured_fps, last_latency_ms, frame_index, recording, mask_generator is not None, last_depth_raw_window, depth_window)
                elif key == 27:
                    break

            if recording:
                if writer is None:
                    writer_info = make_writer(record_path, source_fps, (view.shape[1], view.shape[0]))
                    if writer_info is not None:
                        writer, active_record_path = writer_info
                if writer is not None:
                    writer.write(view)

            if args.print_every > 0 and frame_index % args.print_every == 0:
                elapsed = max(time.time() - last_report, 1e-6)
                last_report = time.time()
                state = last_result.smoothed_state if last_result else "Waiting"
                print(
                    f"[camera] frame={frame_index} state={state} "
                    f"capture_fps={args.print_every / elapsed:.2f} infer_ms={last_latency_ms or 0.0:.1f}"
                )
    except KeyboardInterrupt:
        print("\n[camera] interrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if jsonl_file is not None:
            jsonl_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    if writer is not None:
        print(f"[camera] wrote annotated recording: {active_record_path}")


if __name__ == "__main__":
    main()
