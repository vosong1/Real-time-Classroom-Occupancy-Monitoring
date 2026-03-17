from pathlib import Path
import random

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
COCO_TRAIN_LIMIT = 5000
COCO_VAL_LIMIT = 1000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "dataset"


def collect_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def image_to_label(img_path: Path) -> Path:
    rel = img_path.relative_to(DATASET / "images")
    return (DATASET / "labels" / rel).with_suffix(".txt")


def valid_images(folder: Path):
    imgs = collect_images(folder)
    out = []
    for img in imgs:
        lbl = image_to_label(img)
        if img.exists() and lbl.exists():
            out.append(img)
    return out


def sample_paths(paths, limit):
    if len(paths) <= limit:
        return paths
    rng = random.Random(SEED)
    paths = paths.copy()
    rng.shuffle(paths)
    return sorted(paths[:limit])


def write_list(paths, out_file: Path):
    with open(out_file, "w", encoding="utf-8") as f:
        for p in paths:
            rel = p.relative_to(PROJECT_ROOT)
            f.write(rel.as_posix() + "\n")


def main():
    train_classrooms = valid_images(DATASET / "images" / "train" / "classrooms")
    train_coco = valid_images(DATASET / "images" / "train" / "coco")
    val_classrooms = valid_images(DATASET / "images" / "val" / "classrooms")
    val_coco = valid_images(DATASET / "images" / "val" / "coco")

    train_coco = sample_paths(train_coco, COCO_TRAIN_LIMIT)
    val_coco = sample_paths(val_coco, COCO_VAL_LIMIT)

    train_final = train_classrooms * 5 + train_coco
    val_final = val_classrooms + val_coco

    write_list(train_final, DATASET / "train.txt")
    write_list(val_final, DATASET / "val.txt")

    print("train_classrooms:", len(train_classrooms))
    print("train_coco:", len(train_coco))
    print("train_final:", len(train_final))
    print("val_classrooms:", len(val_classrooms))
    print("val_coco:", len(val_coco))
    print("val_final:", len(val_final))


if __name__ == "__main__":
    main()