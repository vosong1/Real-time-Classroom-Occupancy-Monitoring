from ultralytics import YOLO

def main():
    model = YOLO("yolov3u.pt")

    model.train(
        data="D:/Real-time-Classroom-Occupancy-Monitoring/dataset/data.yaml",
        imgsz=416,
        batch=4,
        epochs=50
    )

if __name__ == "__main__":
    main()