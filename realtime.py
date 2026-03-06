import cv2
from ultralytics import YOLO

MODEL_PATH = r"runs/detect/train/weights/best.pt"
PERSON_CLASS_ID = 0
CONF = 0.25
THRESHOLD = 30  # tuỳ lớp

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(frame, conf=CONF, verbose=False)[0]

    count_person = 0
    for b in results.boxes:
        cls = int(b.cls[0])
        if cls == PERSON_CLASS_ID:
            count_person += 1

    if count_person == 0:
        status = "EMPTY"
    elif count_person > THRESHOLD:
        status = "OVERCROWDED"
    else:
        status = "OCCUPIED"

    cv2.putText(frame, f"Persons: {count_person} | {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Occupancy", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()