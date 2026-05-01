import torch
from torchvision import transforms
from PIL import Image
import os

from .model import CNNModel

device = "cuda" if torch.cuda.is_available() else "cpu"

classes = sorted([
    c for c in os.listdir("data/spectrograms/train")
    if os.path.isdir(os.path.join("data/spectrograms/train", c))
])

model = CNNModel().to(device)
model.load_state_dict(torch.load("model/cnn/model.pth", map_location=device,weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _,pred = torch.max(model(img),1)

    return classes[pred.item()]


# import torch
# from torchvision import transforms
# from PIL import Image

# from .model import CNNModel

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load class names from file
# with open("model/cnn/classes.txt") as f:
#     classes = [line.strip() for line in f]

# model = CNNModel().to(device)
# model.load_state_dict(torch.load("model/cnn/model.pth", map_location=device))
# model.eval()

# transform = transforms.Compose([
#     transforms.Resize((128,128)),
#     transforms.ToTensor()
# ])

# def predict_image(path):
#     img = Image.open(path).convert("RGB")
#     img = transform(img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         _, pred = torch.max(model(img), 1)

#     return classes[pred.item()]