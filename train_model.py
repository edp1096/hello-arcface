import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchinfo import summary
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor

import modules.dataset as dset
import modules.fit as fit
import modules.valid as valid
import modules.arcface as af
import modules.ccface as cc

import random


use_torchvision_dataset = False
# model_fname = "model_mnist_default.pt"
model_fname = "model_mnist_arcface.pt"
# model_fname = "model_mnist_ccface.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

random.seed(777)
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

# epochs = 5
# batch_size = 100
# learning_rate = 0.01
epochs = 17
batch_size = 512
learning_rate = 0.05

num_classes = 10

train_transform = transforms.Compose([transforms.ToTensor()])
valid_transform = train_transform

if use_torchvision_dataset:
    train_set, valid_set = dset.prepareTorchvisionDataset(train_transform, valid_transform)  # Torchvision Dataset
else:
    train_set = dset.prepareCustomDataset(9, "datas/train", train_transform)  # Custom Dataset
    valid_set = dset.prepareCustomDataset(9, "datas/test", valid_transform)

train_loader, valid_loader = dset.getDataLoaders(train_set, valid_set, batch_size, batch_size)

# resnet
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
# model.fc = nn.Linear(model.fc.in_features, num_classes)  # resnet
model.fc = af.ArcFace(model.fc.in_features, num_classes)
# model.fc = af.ArcMarginProduct(model.fc.in_features, num_classes)
# model.fc = cc.CurricularFace(model.fc.in_features, num_classes)

model.to(device)
print(model)

summary(model, input_size=(512, 1, 28, 28))  # cnn

total_batch = len(train_loader)
print("Batch count : {}".format(total_batch))

criterion = nn.CrossEntropyLoss().to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 훈련 시작
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    # cnn
    fit.runCNN(device, train_loader, model, criterion, optimizer)
    valid.runCNN(device, valid_loader, model, criterion)

print("Training done!")

torch.save(model.state_dict(), model_fname)
print(f"Saved PyTorch Model State to {model_fname}")
