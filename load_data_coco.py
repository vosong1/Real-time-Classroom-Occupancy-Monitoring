from pathlib import Path
import random
import shutil

RANDOM_SEED = 42
NUM_IMAGES = 5000

SRC_IMAGES = Path("dataset/images/train")
SRC_LABELS = Path("dataset/labels/train")

DST_IMAGES = Path("dataset_5k/images/train")
DST_LABELS = Path("dataset_5k/labels/train")

IMG_EXTS = {".jpg", ".jpeg", ".png"}

# class ids sau khi remap YOLO
# 0 = person, 1 = backpack, 2 = chair
TARGET_CLASS_IDS = {0, 1, 2}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def get_label_path(img_path: Path, src_images_root: Path, src_labels_root: Path) -> Path:
    rel_path = img_path.relative_to(src_images_root)
    return (src_labels_root / rel_path).with_suffix(".txt")


def label_has_target_class(label_path: Path) -> bool:
    if not label_path.exists():
        return False

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return False

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue

        if cls_id in TARGET_CLASS_IDS:
            return True

    return False


def collect_valid_images(src_images_root: Path, src_labels_root: Path) -> list[Path]:
    valid_images = []

    for img_path in src_images_root.rglob("*"):
        if not is_image_file(img_path):
            continue

        label_path = get_label_path(img_path, src_images_root, src_labels_root)

        if label_has_target_class(label_path):
            valid_images.append(img_path)

    return sorted(valid_images)


def copy_selected_dataset(selected_images: list[Path]) -> None:
    for img_path in selected_images:
        rel_path = img_path.relative_to(SRC_IMAGES)

        src_label_path = get_label_path(img_path, SRC_IMAGES, SRC_LABELS)

        dst_img_path = DST_IMAGES / rel_path
        dst_label_path = (DST_LABELS / rel_path).with_suffix(".txt")

        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, dst_img_path)
        shutil.copy2(src_label_path, dst_label_path)


def main() -> None:
    random.seed(RANDOM_SEED)

    if not SRC_IMAGES.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc anh: {SRC_IMAGES}")

    if not SRC_LABELS.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc label: {SRC_LABELS}")

    valid_images = collect_valid_images(SRC_IMAGES, SRC_LABELS)

    print(f"Tong so anh co person/backpack/chair: {len(valid_images)}")

    if not valid_images:
        raise ValueError("Khong co anh hop le nao de lay mau.")

    num_to_sample = min(NUM_IMAGES, len(valid_images))
    selected_images = random.sample(valid_images, num_to_sample)

    print(f"Bat dau copy {num_to_sample} anh...")
    copy_selected_dataset(selected_images)

    print("Done.")
    print(f"Dataset moi nam o: {DST_IMAGES.parent.parent}")


if __name__ == "__main__":
    main()