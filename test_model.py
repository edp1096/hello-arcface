import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.io import ImageReadMode
from torchvision.transforms import ToTensor

import modules.dataset as dset
import modules.network as net
import modules.arcface as af
import modules.ccface as cc

import matplotlib.pyplot as plt
import random

# model_fname = "model_mnist_default.pt"
# model_fname = "model_mnist_arcface.pt"
model_fname = "model_mnist_ccface.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# random.seed(777)
# torch.manual_seed(777)
# if device == "cuda":
#     torch.cuda.manual_seed_all(777)

num_classes = 10

test_transform = transforms.Compose([transforms.ToTensor()])
test_data = dset.prepareCustomDataset(9, "datas/test", test_transform)

# resnet
model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
# model.fc = nn.Linear(512, 10) # resnet18
# model.fc = nn.Linear(2048, 10)  # resnet50
# model.fc = af.ArcFace(model.fc.in_features, num_classes)
# model.fc = af.ArcMarginProduct(model.fc.in_features, num_classes)
model.fc = cc.CurricularFace(model.fc.in_features, num_classes)

model.to(device)
model.load_state_dict(torch.load(model_fname))
model.eval()

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
r = random.randint(0, len(test_data) - 1)

# image, label = test_data[r][0], test_data[r][1]
image, label = test_data[r]

# 임의 숫자
# from torchvision.io import read_image
# fname, label = "5.jpg", 5
# image = read_image(fname, ImageReadMode.GRAY)
# image = transforms.Resize((28, 28))(image)

with torch.no_grad():
    pred = model(image.float().unsqueeze(dim=0).to(device))  # cnn

predicted, actual = classes[pred[0].argmax(0).cpu().numpy()], classes[label]
probability = (255 - pred[0].cpu().numpy()[predicted]) / 255

# print(pred, classes, pred[0].argmax(0).cpu().numpy())

probability_pct = f"{probability * 100:.2f}"
tensor_value = f"{pred[0].cpu().numpy()[predicted]:.2f}"

print(f"Index: {r}, Tensor value: {tensor_value}")
print(f"Probability(%): {probability_pct}%, Predicted: {predicted}, Actual: {actual}")

plt.title(f"Predicted: {predicted}, Actual: {actual}")
plt.imshow(image.view(28, 28).cpu().numpy(), cmap="Greys", interpolation="nearest")
plt.show()
