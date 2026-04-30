from ultralytics import YOLO
import os

model = YOLO("runs/detect/train/weights/best.pt")

folder = "underwater_yolo/test/images"
img = os.listdir(folder)[0]

results = model.predict(f"{folder}/{img}", save=True)

for r in results:
    print(r.names)