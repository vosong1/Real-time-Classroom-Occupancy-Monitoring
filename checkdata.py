from pathlib import Path

DATASET_ROOT = Path("dataset")
IMG_EXTS = {".jpg", ".jpeg", ".png"}
NUM_CLASSES = 1


def collect_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def collect_labels(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.txt"))


def validate_label_file(label_path: Path, num_classes: int) -> list[str]:
    errors = []

    try:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as exc:
        return [f"{label_path}: khong doc duoc file ({exc})"]

    for line_no, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"{label_path} line {line_no}: phai co 5 gia tri")
            continue

        try:
            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{label_path} line {line_no}: khong parse duoc so")
            continue

        if not (0 <= cls_id < num_classes):
            errors.append(f"{label_path} line {line_no}: class_id khong hop le ({cls_id})")

        for name, val in zip(["x", "y", "w", "h"], [x, y, w, h]):
            if not (0.0 <= val <= 1.0):
                errors.append(f"{label_path} line {line_no}: {name} nam ngoai [0,1] ({val})")

    return errors


def check_split(split: str) -> None:
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split

    images = collect_images(img_dir)
    labels = collect_labels(lbl_dir)

    image_stems = {p.relative_to(img_dir).with_suffix("") for p in images}
    label_stems = {p.relative_to(lbl_dir).with_suffix("") for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    print(f"\n=== SPLIT: {split} ===")
    print(f"So anh   : {len(images)}")
    print(f"So label : {len(labels)}")
    print(f"Anh thieu label : {len(missing_labels)}")
    print(f"Label thieu anh : {len(missing_images)}")

    if missing_labels[:10]:
        print("Vi du anh thieu label:")
        for x in missing_labels[:10]:
            print(" -", x)

    if missing_images[:10]:
        print("Vi du label thieu anh:")
        for x in missing_images[:10]:
            print(" -", x)

    all_errors = []
    for label_path in labels:
        all_errors.extend(validate_label_file(label_path, NUM_CLASSES))

    print(f"So loi format label: {len(all_errors)}")
    for err in all_errors[:20]:
        print(" -", err)


def main() -> None:
    check_split("train")
    check_split("val")


if __name__ == "__main__":
    main()