from pathlib import Path
import shutil

COCO_ROOT = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\coco")
CLASS_ROOT = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\classroom")
OUT_ROOT = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\dataset")

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def copy_split(src_root: Path, split: str, source_name: str):
    src_img = src_root / "images" / split
    src_lbl = src_root / "labels" / split

    dst_img = OUT_ROOT / "images" / split / source_name
    dst_lbl = OUT_ROOT / "labels" / split / source_name

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for img_path in src_img.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        label_path = src_lbl / f"{img_path.stem}.txt"

        shutil.copy2(img_path, dst_img / img_path.name)

        if label_path.exists():
            shutil.copy2(label_path, dst_lbl / label_path.name)

def write_yaml():
    yaml_path = OUT_ROOT / "data.yaml"
    yaml_text = """path: .
train: images/train
val: images/val

names:
  0: person
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print("[OK] wrote:", yaml_path)

if __name__ == "__main__":
    copy_split(COCO_ROOT, "train", "coco")
    copy_split(COCO_ROOT, "val", "coco")

    copy_split(CLASS_ROOT, "train", "classroom")
    copy_split(CLASS_ROOT, "val", "classroom")

    write_yaml()
    print("[DONE] merged dataset ->", OUT_ROOT)