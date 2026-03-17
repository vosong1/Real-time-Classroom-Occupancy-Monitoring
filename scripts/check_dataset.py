from pathlib import Path

images = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\dataset\images")
labels = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\dataset\labels")

exts = {".jpg",".jpeg",".png",".bmp",".webp"}

missing = []

for img in images.rglob("*"):
    if img.suffix.lower() in exts:
        lbl = labels / img.relative_to(images).with_suffix(".txt")
        if not lbl.exists():
            missing.append(img)

print("Missing:",len(missing))

for m in missing:
    print(m)