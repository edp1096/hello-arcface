from config import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import modules.arcface as af
import modules.ccface as cc
import modules.xfrm as xfrm

import matplotlib.pyplot as plt
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# data_transform = transforms.ToTensor()
data_transform = transforms.Compose(
    [
        xfrm.CLAHE(clipLimit=2.5),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)
test_set = datasets.ImageFolder(f"{DATA_ROOT}/valid", transform=data_transform)

num_classes = len(test_set.classes)

match MODEL_NAME:
    case "resnet50":
        model = models.resnet50()

match FC_LAYER:
    case "default":
        model.fc = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(model.fc.in_features, num_classes))

    case "arcface":
        model.fc = nn.Sequential(nn.Dropout(p=0.4, inplace=True), af.ArcFace(model.fc.in_features, num_classes))

    case "ccface":
        model.fc = nn.Sequential(nn.Dropout(p=0.4, inplace=True), cc.CurricularFace(model.fc.in_features, num_classes))

model.to(device)
model.load_state_dict(torch.load(WEIGHT_FILENAME))
model.eval()

r = random.randint(0, len(test_set) - 1)  # 랜덤 숫자
image, label_idx = test_set[r]  # image, label_idx = test_data[r][0], test_data[r][1]

with torch.no_grad():
    pred = model(image.float().unsqueeze(dim=0).to(device))


# 라벨
classes = test_set.classes
class_idx = test_set.class_to_idx

pred_max_idx = pred[0].argmax(0).cpu().numpy()
pred_max_value = pred[0].max().cpu().numpy()
predicted, actual = classes[pred_max_idx], classes[label_idx]
probability = (255 - pred_max_value) / 255
probability_pct = f"{probability * 100:.2f}"

print(f"Index: {r}, Tensor value: {pred_max_value:.2f}")
print(f"Probability(%): {probability_pct}%, Predicted: {predicted}, Actual: {actual}")

plt.title(f"Predicted: {predicted}, Actual: {actual}")
plt.imshow(image.permute(1, 2, 0).cpu().numpy())

plt.show()
