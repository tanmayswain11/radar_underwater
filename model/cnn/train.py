import torch
from torchvision import datasets, transforms
from model import CNNModel   # 🔥 FIXED
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# ---------------- DATASET ----------------
dataset = datasets.ImageFolder("data/spectrograms/train", transform=transform)

print("✅ Classes:", dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(dataset,[train_size,val_size])

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=8,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds,batch_size=8)

# ---------------- MODEL ----------------
model = CNNModel().to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss()

best_acc = 0
loss_list = []
acc_list = []

# ---------------- TRAIN ----------------
for epoch in range(10):
    model.train()
    running_loss = 0

    for x,y in train_loader:
        x,y = x.to(device), y.to(device)

        output = model(x)
        loss = loss_fn(output,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # -------- VALIDATION --------
    correct,total = 0,0
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)

            output = model(x)
            _,pred = torch.max(output,1)

            total += y.size(0)
            correct += (pred==y).sum().item()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = 100*correct/total
    loss_list.append(running_loss)
    acc_list.append(acc)

    print(f"Epoch {epoch+1}: Loss={running_loss:.2f}, Acc={acc:.2f}%")

    # SAVE BEST MODEL
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(),"model/cnn/model.pth")

# ---------------- METRICS ----------------
cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')

os.makedirs("metrics", exist_ok=True)

with open("metrics/metrics.json","w") as f:
    json.dump({
        "accuracy": acc_list,
        "loss": loss_list,
        "confusion_matrix": cm.tolist(),
        "precision": float(precision),
        "recall": float(recall),
        "classes": dataset.classes
    }, f)

print("🔥 Best Accuracy:", best_acc)
print("✅ Metrics Saved")