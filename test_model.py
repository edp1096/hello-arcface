from config import *
from common import device, data_transform

import torch
from torchvision import datasets, models, transforms

from modules.net import NetHead
from modules.file import loadWeights

import matplotlib.pyplot as plt
import random


test_set = datasets.ImageFolder(f"{DATA_ROOT}/valid", transform=data_transform)

num_classes = len(test_set.classes)

model = NetHead(num_classes)
model.to(device)
model.load_state_dict(loadWeights()["model"])
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

print(pred_max_idx, pred_max_value)

predicted, actual = classes[pred_max_idx], classes[label_idx]
probability = (255 - pred_max_value) / 255
probability_pct = f"{probability * 100:.2f}"

print(f"Index: {r}, Tensor value: {pred_max_value:.2f}")
print(f"Probability(%): {probability_pct}%, Predicted: {predicted}, Actual: {actual}")

plt.title(f"Predicted: {predicted}, Actual: {actual}")
plt.imshow(image.permute(1, 2, 0).cpu().numpy())

plt.show()
