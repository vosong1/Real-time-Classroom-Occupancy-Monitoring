from pathlib import Path

src = Path("configs/yolov3.cfg")
dst = Path("configs/yolov3-classroom.cfg")

text = src.read_text(encoding="utf-8")

# sửa phần đầu
text = text.replace("batch=1", "batch=64", 1)
text = text.replace("subdivisions=1", "subdivisions=16", 1)
text = text.replace("max_batches = 500200", "max_batches=6000")
text = text.replace("steps=400000,450000", "steps=4800,5400")

# sửa classes
text = text.replace("classes=80", "classes=3")

# sửa filters trước yolo
count = 0
parts = text.split("[yolo]")
new_parts = [parts[0]]

for part in parts[1:]:
    head = new_parts[-1]
    idx = head.rfind("filters=255")
    if idx != -1:
        head = head[:idx] + "filters=24" + head[idx + len("filters=255"):]
        new_parts[-1] = head
        count += 1
    new_parts.append("[yolo]" + part)

fixed = "".join(new_parts)
dst.write_text(fixed, encoding="utf-8")

print(f"Done. Wrote: {dst}")
print(f"Updated filters sections: {count}")