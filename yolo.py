from pathlib import Path
DATASET_ROOT = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\dataset_yolo")
OUT_DATA_DIR = Path(r"D:\Real-time-Classroom-Occupancy-Monitoring\data") 
NAMES = ["person", "chair", "backpack"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])
def main():
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_imgs = list_images(DATASET_ROOT / "images" / "train")
    val_imgs   = list_images(DATASET_ROOT / "images" / "val")
    (OUT_DATA_DIR / "obj.names").write_text("\n".join(NAMES) + "\n", encoding="utf-8")
    def to_darknet_path(p: Path) -> str:
        return str(p).replace("\\", "/")

    (OUT_DATA_DIR / "train.txt").write_text(
        "\n".join(to_darknet_path(p) for p in train_imgs) + "\n", encoding="utf-8"
    )
    (OUT_DATA_DIR / "val.txt").write_text(
        "\n".join(to_darknet_path(p) for p in val_imgs) + "\n", encoding="utf-8"
    )

    # 3) obj.data
    # weights
    backup_dir = OUT_DATA_DIR.parent / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    obj_data = f"""classes = {len(NAMES)}
train  = {to_darknet_path(OUT_DATA_DIR / "train.txt")}
valid  = {to_darknet_path(OUT_DATA_DIR / "val.txt")}
names  = {to_darknet_path(OUT_DATA_DIR / "obj.names")}
backup = {to_darknet_path(backup_dir)}
"""
    (OUT_DATA_DIR / "obj.data").write_text(obj_data, encoding="utf-8")
    print("DONE ")
    print("Wrote:", OUT_DATA_DIR / "obj.names")
    print("Wrote:", OUT_DATA_DIR / "obj.data")
    print("Wrote:", OUT_DATA_DIR / "train.txt")
    print("Wrote:", OUT_DATA_DIR / "val.txt")
    print("Backup folder:", backup_dir)

if __name__ == "__main__":
    main()