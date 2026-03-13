from pathlib import Path
import yaml

DATASET_ROOT = Path("dataset")
CLASS_NAMES = ["person"]


def count_files(folder: Path, exts: set[str]) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def create_data_yaml(dataset_root: Path, class_names: list[str]) -> Path:
    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    return yaml_path


def main() -> None:
    img_exts = {".jpg", ".jpeg", ".png"}

    train_img_dir = DATASET_ROOT / "images" / "train"
    val_img_dir = DATASET_ROOT / "images" / "val"
    train_lbl_dir = DATASET_ROOT / "labels" / "train"
    val_lbl_dir = DATASET_ROOT / "labels" / "val"

    for folder in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if not folder.exists():
            raise FileNotFoundError(f"Khong tim thay thu muc: {folder}")

    print("Train images:", count_files(train_img_dir, img_exts))
    print("Val images  :", count_files(val_img_dir, img_exts))
    print("Train labels:", sum(1 for _ in train_lbl_dir.rglob("*.txt")))
    print("Val labels  :", sum(1 for _ in val_lbl_dir.rglob("*.txt")))

    yaml_path = create_data_yaml(DATASET_ROOT, CLASS_NAMES)
    print(f"Da tao: {yaml_path}")


if __name__ == "__main__":
    main()