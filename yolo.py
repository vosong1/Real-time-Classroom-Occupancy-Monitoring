from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
DATA_YAML = "dataset/data.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 16
PROJECT = "runs/detect"
NAME = "classroom_person"


def main() -> None:
    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        pretrained=True,
    )


if __name__ == "__main__":
    main()