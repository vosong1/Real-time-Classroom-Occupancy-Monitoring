import time
import cv2
import torch

MODEL_PATH = "run/best.pt"

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    force_reload=False
)

model.conf = 0.4

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    labels = results.xyxy[0]

    person_count = 0

    for *box, conf, cls in labels:
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)
        label_name = model.names[cls]

        if label_name == "person":
            person_count += 1
            color = (0, 255, 0)
        elif label_name == "chair":
            color = (255, 200, 0)
        elif label_name in ["bag", "backpack"]:
            color = (255, 0, 255)
        else:
            color = (200, 200, 200)

        label = f"{label_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    cv2.rectangle(frame, (10, 10), (330, 90), (30, 30, 30), -1)
    cv2.putText(frame, f"People: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Classroom Occupancy Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()