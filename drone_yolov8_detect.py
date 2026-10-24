"""
Drone Camera Object Detection — YOLOv8n
-----------------------------------------
Detects humans and vehicles (car, motorcycle, bus, truck, bicycle) in drone
video footage (file, folder of files, RTSP/UDP stream, or webcam index).

Install:
    pip install ultralytics opencv-python --break-system-packages

Usage:
    python drone_yolov8_detect.py --source input_video.mp4 --output out.mp4
    python drone_yolov8_detect.py --source rtsp://<drone-stream-url>
    python drone_yolov8_detect.py --source 0                     # webcam
    python drone_yolov8_detect.py --source drone_clips/ --output_dir results/
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class IDs relevant to drone surveillance / SAR use cases.
# person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8n detection for drone footage")
    p.add_argument("--source", required=True,
                    help="Video file, folder of videos, RTSP/UDP URL, or webcam index (0,1,...)")
    p.add_argument("--weights", default="yolov8n.pt",
                    help="Model weights (default: yolov8n.pt, auto-downloads on first run)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0-1)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--device", default="", help="'' auto, 'cpu', '0' for GPU 0, etc.")
    p.add_argument("--output", default=None, help="Output annotated video path (single-file mode)")
    p.add_argument("--output_dir", default="drone_detections",
                    help="Output directory (used for folder mode / logs)")
    p.add_argument("--track", action="store_true",
                    help="Enable multi-object tracking (persistent IDs) instead of per-frame detection")
    p.add_argument("--classes", nargs="*", type=int, default=list(TARGET_CLASSES.keys()),
                    help="COCO class IDs to keep (default: person + vehicle classes)")
    p.add_argument("--log_csv", default=None, help="Path to write per-detection CSV log")
    p.add_argument("--no_display", action="store_true", help="Disable live preview window")
    return p.parse_args()


def open_source(source):
    """Return an int (webcam), or the raw string (file/stream) for cv2/YOLO to consume."""
    if source.isdigit():
        return int(source)
    return source


def annotate_frame(frame, boxes, names):
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = names.get(cls_id, str(cls_id))
        track_id = int(box.id[0]) if box.id is not None else None
        text = f"{label} {conf:.2f}" + (f" ID:{track_id}" if track_id is not None else "")

        color = (0, 0, 255) if cls_id == 0 else (255, 140, 0)  # red=person, orange=vehicle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def process_video(model, source, args, out_path, csv_writer=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        predict_fn = model.track if args.track else model.predict
        kwargs = dict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=args.classes,
            device=args.device or None,
            verbose=False,
        )
        if args.track:
            kwargs["persist"] = True

        results = predict_fn(**kwargs)
        r = results[0]
        names = r.names

        frame = annotate_frame(frame, r.boxes, names)

        if csv_writer is not None:
            timestamp_s = frame_idx / fps
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else ""
                csv_writer.writerow([
                    frame_idx, f"{timestamp_s:.2f}", names.get(cls_id, cls_id),
                    f"{conf:.3f}", track_id, x1, y1, x2, y2
                ])

        if writer:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Drone Detection (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - t_start
    print(f"[INFO] Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.1f} FPS) -> {source}")

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading model: {args.weights}")
    model = YOLO(args.weights)

    csv_file = None
    csv_writer = None
    log_path = args.log_csv or os.path.join(args.output_dir, "detections_log.csv")
    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "time_s", "class", "confidence", "track_id",
                          "x1", "y1", "x2", "y2"])

    source_path = Path(args.source)

    try:
        if source_path.exists() and source_path.is_dir():
            video_files = sorted([
                f for f in source_path.iterdir()
                if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
            ])
            if not video_files:
                print(f"[ERROR] No video files found in {source_path}", file=sys.stderr)
                sys.exit(1)
            for vf in video_files:
                out_path = os.path.join(args.output_dir, f"annotated_{vf.stem}.mp4")
                print(f"[INFO] Processing {vf} -> {out_path}")
                process_video(model, str(vf), args, out_path, csv_writer)
        else:
            src = open_source(args.source)
            out_path = args.output or os.path.join(args.output_dir, "annotated_output.mp4")
            process_video(model, src, args, out_path, csv_writer)
    finally:
        if csv_file:
            csv_file.close()
            print(f"[INFO] Detection log saved to {log_path}")


if __name__ == "__main__":
    main()
