import json
from pathlib import Path

# =========================
# SỬA 2 PATH NÀY
# =========================
COCO_ROOT = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring")  # nơi có annotations/instances_*.json (COCO gốc)
IMAGES_FILTERED_ROOT = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\dataset_yolo\images")  # nơi có ảnh đã lọc (ví dụ nếu bạn đang dùng dataset_yolo thì đổi thành: Path(r"...\dataset_yolo\images")
# ví dụ nếu bạn đang dùng dataset_yolo thì đổi thành: Path(r"...\dataset_yolo\images")

# Output labels sẽ được tạo song song với images:
# <IMAGES_FILTERED_ROOT>\..\labels\<train|val>
LABELS_OUT_ROOT = IMAGES_FILTERED_ROOT.parent / "labels"

# COCO category_id -> YOLO class_id
# person=1, chair=62, backpack=27
COCO_TO_YOLO = {1: 0, 62: 1, 27: 2}

def coco_bbox_to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    xc = (x + bw / 2) / w
    yc = (y + bh / 2) / h
    return xc, yc, bw / w, bh / h

def build_needed_filenames(img_dir: Path):
    """Lấy danh sách file ảnh đang có trong folder đã lọc"""
    exts = {".jpg", ".jpeg", ".png"}
    return {p.name for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}

def regen_split(split: str):
    assert split in {"train", "val"}

    ann_path = COCO_ROOT / "annotations" / f"instances_{split}2017.json"
    img_dir_filtered = IMAGES_FILTERED_ROOT / split
    out_dir = LABELS_OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.exists():
        raise FileNotFoundError(f"Không thấy COCO JSON gốc: {ann_path}")
    if not img_dir_filtered.exists():
        raise FileNotFoundError(f"Không thấy folder ảnh đã lọc: {img_dir_filtered}")

    needed_files = build_needed_filenames(img_dir_filtered)
    print(f"[{split}] Found filtered images: {len(needed_files)}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))

    # image_id -> image_info
    imgs = {im["id"]: im for im in data["images"]}

    # Chỉ giữ image_id có file_name nằm trong filtered folder
    keep_img_ids = {im_id for im_id, im in imgs.items() if im["file_name"] in needed_files}

    # Gom annotations theo image_id và chỉ lấy 3 class
    anns_by_img = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in keep_img_ids:
            continue
        cid = ann["category_id"]
        if cid not in COCO_TO_YOLO:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    wrote = 0
    empty = 0

    # Tạo label cho từng ảnh đang có (kể cả ảnh không có bbox 3 class -> file rỗng)
    for im_id in keep_img_ids:
        im = imgs[im_id]
        w, h = im["width"], im["height"]
        stem = Path(im["file_name"]).stem
        label_path = out_dir / f"{stem}.txt"

        anns = anns_by_img.get(im_id, [])
        lines = []
        for a in anns:
            cls = COCO_TO_YOLO[a["category_id"]]
            xc, yc, bw, bh = coco_bbox_to_yolo(a["bbox"], w, h)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text("\n".join(lines), encoding="utf-8")
        wrote += 1
        if not lines:
            empty += 1

    print(f"[{split}] Wrote label files: {wrote} (empty: {empty}) -> {out_dir}")

if __name__ == "__main__":
    regen_split("train")
    regen_split("val")
    print("[DONE] Labels regenerated from COCO JSON gốc.")