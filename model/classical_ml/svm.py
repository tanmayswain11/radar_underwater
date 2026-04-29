import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import json

MODEL_PATH = "model/classical_ml/svm_model.pkl"

def train_svm():

    # ---------------- LOAD IF EXISTS ----------------
    if os.path.exists(MODEL_PATH):
        print("⚡ Loading saved SVM...")
        return joblib.load(MODEL_PATH)   # (model, scaler, classes)

    print("🚀 Training SVM (FAST VERSION)...")

    X, y = [], []
    base_path = "data/spectrograms/train"

    # 🔥 SORT CLASSES
    classes = sorted([
        c for c in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, c))
    ])

    print("✅ Classes:", classes)

    # ---------------- LOAD IMAGES ----------------
    count = 0

    for label, cls in enumerate(classes):
        cls_path = os.path.join(base_path, cls)

        for root, _, files in os.walk(cls_path):
            for img in files:

                if not img.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                img_path = os.path.join(root, img)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                try:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (32,32)).flatten()

                    X.append(image)
                    y.append(label)

                    count += 1
                    if count % 2000 == 0:
                        print(f"📥 Loaded {count} images...")

                except:
                    continue

    # ---------------- CHECK DATA ----------------
    if len(X) == 0:
        raise ValueError("❌ No images found! Check dataset path.")

    X = np.array(X)
    y = np.array(y)

    print("📊 Total samples:", len(X))

    # ---------------- SCALE DATA ----------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ---------------- TRAIN ----------------
    print("⚡ Training Linear SVM...")
    model = LinearSVC(max_iter=2000)

    model.fit(X, y)

    # ---------------- ACCURACY ----------------
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"🔥 SVM Accuracy: {acc*100:.2f}%")

    # ---------------- SAVE MODEL ----------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((model, scaler, classes), MODEL_PATH)

    # ---------------- SAVE METRICS ----------------
    os.makedirs("metrics", exist_ok=True)

    with open("metrics/svm_metrics.json", "w") as f:
        json.dump({
            "accuracy": float(acc*100)
        }, f)

    print("✅ SVM Saved + Metrics Saved")

    return model, scaler, classes


# 🔥 RUN DIRECTLY
if __name__ == "__main__":
    train_svm()