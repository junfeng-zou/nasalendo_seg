#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from distance_state_classifier.src.config import load_config, resolve_path
from distance_state_classifier.src.predictor import DistanceStatePredictor, Prediction
from distance_state_classifier.src.transforms import apply_crop


STATE_COLORS = {
    "TooFar": (255, 180, 60),
    "Good": (80, 220, 90),
    "TooClose": (60, 80, 255),
    "Invalid": (170, 170, 170),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distance-state inference on a video.")
    parser.add_argument("--config", default="distance_state_classifier/configs/distance_state_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--apply-crop", action="store_true", help="Apply raw-frame crop before inference.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-display", action="store_true", help="Do not open the annotated preview window.")
    parser.add_argument("--show", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--display-crop", action="store_true", help="Show and record the cropped effective field of view.")
    parser.add_argument("--preview-scale", type=float, default=0.5)
    parser.add_argument("--record-video", default=None, help="Annotated video output path used when recording is started.")
    parser.add_argument("--record-from-start", action="store_true", help="Record the annotated video from the first processed frame.")
    parser.add_argument("--print-every", type=int, default=30)
    return parser.parse_args()


def default_record_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_recorded.mp4")


def make_writer(path: str | Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    fourcc_name = "mp4v" if suffix in {".mp4", ".m4v"} else "XVID"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc_name), max(fps, 1.0), frame_size)
    if not writer.isOpened():
        raise SystemExit(f"Cannot open video writer: {output_path}")
    return writer


def format_probabilities(result: Prediction) -> str:
    parts = [f"{label}:{prob:.2f}" for label, prob in result.probabilities.items()]
    return "  ".join(parts)


def draw_overlay(
    frame,
    result: Prediction,
    measured_fps: float,
    latency_ms: float,
    frame_index: int,
    recording: bool,
) -> None:
    state = result.smoothed_state
    color = STATE_COLORS.get(state, (255, 255, 255))
    lines = [
        f"Distance: {state}",
        f"raw={result.raw_label} conf={result.raw_confidence:.3f} state={result.state}",
        f"fps={measured_fps:.1f} infer={latency_ms:.1f} ms frame={frame_index}",
        format_probabilities(result),
        "REC: ON  q=stop  Esc=quit" if recording else "REC: OFF  q=start  Esc=quit",
    ]

    x, y = 18, 34
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    line_h = 31
    box_w = min(frame.shape[1] - 20, 840)
    box_h = 18 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    for idx, line in enumerate(lines):
        line_color = color if idx == 0 else (245, 245, 245)
        cv2.putText(frame, line, (x, y + idx * line_h), font, font_scale, line_color, thickness, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (frame.shape[1] - 34, 34), 10, (0, 0, 255), -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    predictor = DistanceStatePredictor(args.checkpoint, config, device=args.device)

    video_path = resolve_path(args.video)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    stride = max(1, int(args.stride))
    record_fps = fps / stride if fps > 0 else 30.0
    record_path = resolve_path(args.record_video) if args.record_video else default_record_path(output_path)

    display = not args.no_display
    if args.record_video and not display and not args.record_from_start:
        print("[infer] --record-video is set, but recording is started by q. Remove --no-display or add --record-from-start.")
    if display:
        print("[infer] preview enabled. Press q to start/stop recording, Esc to quit.")
        print(f"[infer] recording output: {record_path}")

    frame_index = 0
    written = 0
    started = time.time()
    recent_times: list[float] = []
    writer = None
    recording = bool(args.record_from_start)
    if recording:
        print(f"[infer] recording from start: {record_path}")

    try:
        with output_path.open("w", encoding="utf-8") as f:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % stride != 0:
                    frame_index += 1
                    continue
                if args.max_frames is not None and written >= args.max_frames:
                    break

                now = time.time()
                recent_times.append(now)
                recent_times = recent_times[-30:]
                if len(recent_times) >= 2:
                    measured_fps = (len(recent_times) - 1) / max(recent_times[-1] - recent_times[0], 1e-6)
                else:
                    measured_fps = 0.0

                t0 = time.time()
                result = predictor.predict_array(frame, apply_raw_crop=args.apply_crop)
                latency_ms = (time.time() - t0) * 1000.0
                row = {
                    "frame_index": frame_index,
                    "time_sec": frame_index / fps if fps > 0 else None,
                    "raw_label": result.raw_label,
                    "raw_confidence": result.raw_confidence,
                    "state": result.state,
                    "smoothed_state": result.smoothed_state,
                    "probabilities": result.probabilities,
                    "latency_ms": latency_ms,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

                base_view = frame
                if args.display_crop:
                    base_view = apply_crop(base_view, predictor.crop)
                base_view = base_view.copy()
                view = base_view.copy()
                draw_overlay(view, result, measured_fps, latency_ms, frame_index, recording)

                if display:
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
                    cv2.imshow("Distance State Video Inference", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        recording = not recording
                        state_text = "started" if recording else "stopped"
                        print(f"[infer] recording {state_text}: {record_path}")
                        view = base_view.copy()
                        draw_overlay(view, result, measured_fps, latency_ms, frame_index, recording)
                    elif key == 27:
                        break

                if recording:
                    if writer is None:
                        writer = make_writer(record_path, record_fps, (view.shape[1], view.shape[0]))
                    writer.write(view)

                if args.print_every > 0 and written % args.print_every == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    print(f"[infer] frames={written} speed={written / elapsed:.2f} fps state={result.smoothed_state}")
                frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    print(f"[infer] wrote {written} predictions: {output_path}")
    if writer is not None:
        print(f"[infer] wrote annotated recording: {record_path}")


if __name__ == "__main__":
    main()
