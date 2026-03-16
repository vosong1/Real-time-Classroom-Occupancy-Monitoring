from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_to_label_path(img_path: Path):
    parts = list(img_path.parts)

    # dataset/images/train/coco/xxx.jpg
    # -> dataset/labels/train/coco/xxx.txt
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
    except ValueError:
        return None

    return Path(*parts).with_suffix(".txt")


def main():
    root = Path("dataset")
    images = [p for p in (root / "images").rglob("*") if p.suffix.lower() in IMAGE_EXTS]

    missing = []
    empty = []

    for img_path in images:
        label_path = image_to_label_path(img_path)
        if label_path is None or not label_path.exists():
            missing.append((img_path, label_path))
            continue

        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            empty.append(label_path)

    print(f"Total images: {len(images)}")
    print(f"Missing labels: {len(missing)}")
    print(f"Empty labels: {len(empty)}")

    if missing:
        print("\nSome missing labels:")
        for img, lbl in missing[:20]:
            print(f"IMG: {img}")
            print(f"LBL: {lbl}")
            print("-" * 50)

    if empty:
        print("\nSome empty labels:")
        for lbl in empty[:20]:
            print(lbl)


if __name__ == "__main__":
    main()