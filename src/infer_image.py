import os
import cv2
import torch

MODEL_PATH = "run/best.pt"
TEST_FOLDER = "dataset/images/test"
SAVE_FOLDER = "run/test_results"

os.makedirs(SAVE_FOLDER, exist_ok=True)

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    force_reload=False
)
model.conf = 0.4

image_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for img_name in image_files:
    img_path = os.path.join(TEST_FOLDER, img_name)

    results = model(img_path)
    labels = results.xyxy[0]

    person_count = 0

    for *box, conf, cls in labels:
        cls = int(cls)
        name = model.names[cls]

        if name == "person":
            person_count += 1

    img = results.render()[0].copy()

    cv2.putText(
        img,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    save_path = os.path.join(SAVE_FOLDER, img_name)
    cv2.imwrite(save_path, img)

    print(f"{img_name} -> People: {person_count}")

print("Done!")