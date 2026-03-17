import torch
from pathlib import Path
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp8/weights/best.pt', force_reload=True)

img = 'test.jpg'
results = model(img)

labels = results.xyxy[0]

person_count = 0

for *box, conf, cls in labels:
    if int(cls) == 0:  # class 0 = person
        person_count += 1

print("Number of people:", person_count)