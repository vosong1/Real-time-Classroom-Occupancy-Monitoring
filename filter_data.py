import json
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
COCO_ROOT = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\data")          # <-- SỬA THÀNH ĐƯỜNG DẪN CỦA BẠN
SPLIT = "val"                        # "train" hoặc "val"
OUT_ROOT = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\data\filtered")  # <-- thư mục output
KEEP_CAT_IDS = {1, 27, 62}

# Nếu muốn chỉ giữ bbox có area tối thiểu (lọc nhiễu), đặt >0
MIN_BBOX_AREA = 0

# =========================
# PATHS
# =========================
if SPLIT not in {"train", "val"}:
    raise ValueError("SPLIT phải là 'train' hoặc 'val'")

img_dir = COCO_ROOT / (f"{SPLIT}2017")
ann_path = COCO_ROOT / "annotations" / (f"instances_{SPLIT}2017.json")

out_img_dir = OUT_ROOT / "images" / SPLIT
out_ann_dir = OUT_ROOT / "annotations"
out_ann_path = out_ann_dir / (f"instances_{SPLIT}2017_person_chair_backpack.json")

out_img_dir.mkdir(parents=True, exist_ok=True)
out_ann_dir.mkdir(parents=True, exist_ok=True)

if not img_dir.exists():
    raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {img_dir}")
if not ann_path.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file annotation: {ann_path}\n"
        f"Bạn cần giải nén annotations_trainval2017.zip để có instances_{SPLIT}2017.json"
    )

print(f"[INFO] Loading annotation: {ann_path}")
data = json.loads(ann_path.read_text(encoding="utf-8"))

# =========================
# FILTER ANNOTATIONS
# =========================
filtered_annotations = []
keep_image_ids = set()

for ann in data["annotations"]:
    if ann.get("category_id") not in KEEP_CAT_IDS:
        continue
    # bbox = [x, y, w, h]
    bbox = ann.get("bbox", None)
    if bbox and len(bbox) == 4:
        area = bbox[2] * bbox[3]
        if area < MIN_BBOX_AREA:
            continue

    filtered_annotations.append(ann)
    keep_image_ids.add(ann["image_id"])

print(f"[INFO] Kept annotations: {len(filtered_annotations)}")

# =========================
# FILTER IMAGES
# =========================
id_to_img = {img["id"]: img for img in data["images"]}
filtered_images = [id_to_img[iid] for iid in keep_image_ids if iid in id_to_img]
print(f"[INFO] Kept images (from annotations): {len(filtered_images)}")

# =========================
# FILTER CATEGORIES
# =========================
filtered_categories = [c for c in data["categories"] if c["id"] in KEEP_CAT_IDS]

# =========================
# COPY IMAGES
# =========================
copied = 0
missing = 0

print("[INFO] Copying images...")
for img in filtered_images:
    fname = img["file_name"]
    src = img_dir / fname
    dst = out_img_dir / fname
    if src.exists():
        # copy2 giữ metadata
        shutil.copy2(src, dst)
        copied += 1
    else:
        missing += 1

print(f"[INFO] Copied: {copied} | Missing: {missing}")

# =========================
# WRITE NEW JSON
# =========================
out_data = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": filtered_categories,
}

out_ann_path.write_text(json.dumps(out_data, ensure_ascii=False), encoding="utf-8")
print(f"[DONE] Saved filtered annotation JSON: {out_ann_path}")
print(f"[DONE] Output images folder: {out_img_dir}")