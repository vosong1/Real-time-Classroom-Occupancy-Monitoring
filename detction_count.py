from ultralytics import YOLO
import cv2

MODEL_PATH = "runs/detect/classroom_person/weights/best.pt"
IMAGE_PATH = "test/classroom.jpg"
CONF_THRES = 0.4
SAVE_PATH = "output_detect.jpg"


def main() -> None:
    model = YOLO(MODEL_PATH)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {IMAGE_PATH}")

    results = model.predict(source=img, conf=CONF_THRES, verbose=False)

    count_person = 0

    for result in results:
        names = result.names
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            label = names[cls_id]
            if label == "person":
                count_person += 1

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.putText(
        img,
        f"People: {count_person}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )

    print("People count:", count_person)

    cv2.imwrite(SAVE_PATH, img)
    print("Da luu ket qua tai:", SAVE_PATH)

    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()