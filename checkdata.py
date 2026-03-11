from pathlib import Path

DATASET_ROOT = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring")
IMG_EXTS = {".jpg", ".jpeg", ".png"}
NUM_CLASSES = 1

def check_split(split: str):
    img_root = DATASET_ROOT / "images" / split
    lbl_root = DATASET_ROOT / "labels" / split

    all_imgs = [p for p in img_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    all_lbls = [p for p in lbl_root.rglob("*.txt")]

    img_stems = {p.relative_to(img_root).with_suffix("") for p in all_imgs}
    lbl_stems = {p.relative_to(lbl_root).with_suffix("") for p in all_lbls}

    missing_labels = img_stems - lbl_stems
    missing_images = lbl_stems - img_stems

    print(f"\n=== {split.upper()} ===")
    print("images:", len(all_imgs))
    print("labels:", len(all_lbls))
    print("missing labels:", len(missing_labels))
    print("missing images:", len(missing_images))

    bad_format = 0
    bad_class = 0

    for lbl in all_lbls:
        lines = lbl.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                bad_format += 1
                continue

            try:
                cls = int(parts[0])
                vals = list(map(float, parts[1:]))
            except ValueError:
                bad_format += 1
                continue

            if cls < 0 or cls >= NUM_CLASSES:
                bad_class += 1

            for v in vals:
                if v < 0 or v > 1:
                    bad_format += 1
                    break

    print("bad format lines:", bad_format)
    print("bad class lines:", bad_class)

if __name__ == "__main__":
    check_split("train")
    check_split("val")