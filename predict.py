from config import *
from common import device, data_transform

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.transforms import ToPILImage

from modules.net import NetHead
from modules.file import loadWeights

import os
import matplotlib.pyplot as plt
import cv2


classes = os.listdir(f"{DATA_ROOT}/train")
class_idx = {classes[i]: i for i in range(len(classes))}
num_classes = len(classes)


model = NetHead(num_classes)
model.to(device)
model.load_state_dict(loadWeights()["model"])
model.eval()



# IMG_FILENAME = f"{DATA_ROOT}/test/whippet/0..jpg"
IMG_FILENAME = f"{DATA_ROOT}/test/monkey/1.png"

img = cv2.imread(IMG_FILENAME)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

image = ToPILImage()(img)
image = data_transform(image)

with torch.no_grad():
    embed, pred = model(image.float().unsqueeze(dim=0).to(device))


pred_max_idx = pred[0].argmax(0).cpu().numpy()
pred_max_value = pred[0].max().cpu().numpy()
predicted = classes[pred_max_idx]
probability = (255 - pred_max_value) / 255
probability_pct = f"{probability * 100:.2f}"

print(f"Tensor value: {pred_max_value:.2f}, Predicted: {predicted}, Probability: {probability_pct}%")

plt.title(f"Predicted: {predicted}, Probability: {probability_pct}%")
plt.imshow(image.permute(1, 2, 0).cpu().numpy())

plt.show()
