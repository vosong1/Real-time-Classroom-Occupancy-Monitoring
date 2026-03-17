import argparse
from pathlib import Path
import cv2
import torch

def draw_box(image, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_y = max(y1 - 10, th + 6)

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
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=str, required=True, help="Path to image")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--save", type=str, default="runs/detect/result.jpg")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Khong tim thay anh: {source_path}")

    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Khong doc duoc anh: {source_path}")

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

    print("Dang detect...")
    results = model(image)
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
        draw_box(image, (x1, y1, x2, y2), label, color=color)

    empty_est = max(chair_count - person_count, 0)

    summary = (
        f"People: {person_count} | Chairs: {chair_count} | "
        f"Bags: {backpack_count} | Empty est: {empty_est}"
    )

    cv2.rectangle(image, (10, 10), (760, 50), (30, 30, 30), -1)
    cv2.putText(
        image,
        summary,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)

    print(f"Da luu ket qua: {save_path}")

    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()