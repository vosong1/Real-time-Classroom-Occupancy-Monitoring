import os
img_dir = r"D:\Real-time-Classroom-Occupancy-Monitoring\dataset\images\val"
label_dir = r"D:\Real-time-Classroom-Occupancy-Monitoring\dataset\labels\val"
images = sorted(os.listdir(img_dir))
count = 1
for img in images: 
    old_img_path = os.path.join(img_dir, img)  
    if not img.endswith(".jpg"):
        continue
    new_name = f"coco_{count:06d}.jpg"
    new_img_path = os.path.join(img_dir, new_name)
    os.rename(old_img_path, new_img_path)
    old_label = img.replace(".jpg", ".txt")
    old_label_path = os.path.join(label_dir, old_label)

    new_label = new_name.replace(".jpg", ".txt")
    new_label_path = os.path.join(label_dir, new_label)

    if os.path.exists(old_label_path):
        os.rename(old_label_path, new_label_path)

    count += 1

print("Done renaming coco dataset")