from ultralytics import YOLO
import cv2

# sửa đường dẫn best.pt theo đúng runs của bạn
model = YOLO(r"runs\detect\train\weights\best.pt")

cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, verbose=False)

    person = chair = backpack = 0
    for r in results:
        cls = r.boxes.cls
        for c in cls:
            c = int(c)
            if c == 0: person += 1
            elif c == 1: chair += 1
            elif c == 2: backpack += 1

    cv2.putText(frame, f"Person: {person}  Chair: {chair}  Backpack: {backpack}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Count", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()