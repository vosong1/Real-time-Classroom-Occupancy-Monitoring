import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime camera inference on Jetson Orin")
    parser.add_argument("--repo", type=str, default="yolov5", help="Path to local YOLOv5 repo")
    parser.add_argument("--weights", type=str, default="run/best.pt", help="Path to model weights")
    parser.add_argument("--source", type=str, default="0", help="Camera index like 0, or video path")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--save", type=str, default="", help="Optional output video path")
    return parser.parse_args()


def open_source(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def ensure_writer(writer, save_path, frame, cap):
    if not save_path:
        return None
    if writer is not None:
        return writer

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))


def get_device(device_arg: str):
    if device_arg.lower() == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def warmup_model(model, imgsz):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    _ = model(dummy, size=imgsz)


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    repo_path = Path(args.repo).resolve()
    weights_path = Path(args.weights).resolve()

    device = get_device(args.device)

    t0 = time.time()
    model = torch.hub.load(
        str(repo_path),
        "custom",
        path=str(weights_path),
        source="local",
        force_reload=False,
    )
    model.conf = args.conf
    model.iou = 0.45
    model.to(device)

    # Warmup để frame đầu đỡ chậm
    warmup_model(model, args.imgsz)

    load_time = time.time() - t0
    print(f"[INFO] Model loaded on {device} in {load_time:.2f}s")

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    writer = None
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or cannot read frame.")
            break

        results = model(frame, size=args.imgsz)
        labels = results.xyxy[0]

        person_count = 0

        for *box, conf, cls in labels:
            cls = int(cls)
            class_name = model.names[cls]
            if class_name == "person":
                person_count += 1

        rendered = results.render()[0].copy()

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.rectangle(rendered, (10, 10), (320, 100), (30, 30, 30), -1)
        cv2.putText(
            rendered,
            f"People: {person_count}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            rendered,
            f"FPS: {fps:.1f}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Jetson Orin Camera Inference", rendered)

        if args.save:
            writer = ensure_writer(writer, args.save, rendered, cap)
            if writer is not None:
                writer.write(rendered)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()