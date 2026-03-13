from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "runs/detect/classroom_person/weights/best.pt"
CONF_THRES = 0.4
CAMERA_INDEX = 0


def main() -> None:
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc webcam")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Khong doc duoc frame")
            break

        results = model.predict(source=frame, conf=CONF_THRES, verbose=False)

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(frame, f"People: {count_person}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Classroom Occupancy Monitoring", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()