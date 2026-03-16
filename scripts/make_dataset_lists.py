from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def write_list(paths, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p.as_posix() + "\n")


def main():
    root = Path("dataset")

    train_imgs = []
    val_imgs = []

    train_imgs += collect_images(root / "images" / "train" / "classrooms")
    train_imgs += collect_images(root / "images" / "train" / "coco")

    val_imgs += collect_images(root / "images" / "val" / "classrooms")
    val_imgs += collect_images(root / "images" / "val" / "coco")

    write_list(train_imgs, root / "train.txt")
    write_list(val_imgs, root / "val.txt")

    print(f"Train images: {len(train_imgs)}")
    print(f"Val images:   {len(val_imgs)}")
    print("Done.")


if __name__ == "__main__":
    main()