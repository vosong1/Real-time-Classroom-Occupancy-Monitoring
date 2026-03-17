import argparse
from pathlib import Path
import cv2
import torch

def draw_box(image, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_y = max(y1 - 8, th + 6)

    cv2.rectangle(
        image,
        (x1, text_y - th - 6),
        (x1 + tw + 8, text_y + 4),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def open_source(source):
    if source == "0":
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=str, default="0", help="0 = webcam, or video path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--save", type=str, default="")
    args = parser.parse_args()

    print("Dang load model...")
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=args.weights,
        force_reload=False,
    )

    model.conf = args.conf
    model.iou = 0.45
    model.imgsz = args.imgsz

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc source: {args.source}")

    writer = None
    save_video = bool(args.save)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        person_count = 0
        chair_count = 0
        backpack_count = 0

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            cls_name = model.names[cls_id]

            if cls_name == "person":
                person_count += 1
                color = (0, 255, 0)
            elif cls_name == "chair":
                chair_count += 1
                color = (255, 200, 0)
            elif cls_name in ["backpack", "bag"]:
                backpack_count += 1
                color = (255, 0, 255)
            else:
                color = (180, 180, 180)

            label = f"{cls_name} {conf:.2f}"
            draw_box(frame, (x1, y1, x2, y2), label, color=color)

        empty_est = max(chair_count - person_count, 0)

        cv2.rectangle(frame, (10, 10), (760, 55), (30, 30, 30), -1)
        cv2.putText(
            frame,
            f"People: {person_count} | Chairs: {chair_count} | Bags: {backpack_count} | Empty est: {empty_est}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Realtime Detection", frame)

        if save_video:
            if writer is None:
                save_path = Path(args.save)
                save_path.parent.mkdir(parents=True, exist_ok=True)

                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 20.0

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))

            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()