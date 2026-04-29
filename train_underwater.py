from ultralytics import YOLO

if __name__ == "__main__":   # 🔥 MUST ADD THIS

    model = YOLO("yolov8n.pt")

    model.train(
        data="underwater_yolo/data.yaml",
        epochs=30,
        imgsz=416,
        batch=8,
        device=0,
        workers=0   # 🔥 also important for Windows
    )