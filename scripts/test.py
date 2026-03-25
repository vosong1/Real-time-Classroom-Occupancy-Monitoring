import cv2
import torch
import numpy as np
import os
from pathlib import Path

# =========================
# CONFIG (sửa tại đây)
# =========================
MODEL_PATH = "run/best.pt"
TEST_DIR = "dataset/images/test"
GT_FILE = "dataset/gt.txt"

# =========================
# LOAD MODEL YOLOv5
# =========================
model = torch.hub.load("yolov5", "custom", path=MODEL_PATH, source="local")
model.conf = 0.4

# =========================
# LOAD GT
# =========================
gt_dict = {}
with open(GT_FILE, "r") as f:
    for line in f:
        name, count = line.strip().split()
        gt_dict[name] = int(count)

# =========================
# LOOP TEST
# =========================
y_true = []
y_pred = []

image_paths = list(Path(TEST_DIR).glob("*.jpg"))

print("Total images:", len(image_paths))

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    img_name = img_path.name

    results = model(img)
    labels = results.xyxy[0]

    count = 0
    for *box, conf, cls in labels:
        if model.names[int(cls)] == "person":
            count += 1

    if img_name in gt_dict:
        gt = gt_dict[img_name]

        y_true.append(gt)
        y_pred.append(count)

        print(f"{img_name}: GT={gt}, PRED={count}")
    else:
        print(f"⚠️ Missing GT: {img_name}")

# =========================
# CALCULATE MAE
# =========================
if len(y_true) == 0:
    print("❌ No valid data → MAE = NaN")
else:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    print("\nMAE:", round(mae, 2))