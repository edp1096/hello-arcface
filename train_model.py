from config import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import modules.fit as fit
import modules.valid as valid
import modules.arcface as af
import modules.ccface as cc

import matplotlib.pyplot as plt
import random
from torchinfo import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

random.seed(777)
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

# data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_transform = transforms.ToTensor()
train_set = datasets.ImageFolder(f"{DATA_ROOT}/train", transform=data_transform)
valid_set = datasets.ImageFolder(f"{DATA_ROOT}/valid", transform=data_transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(train_set.classes)

match MODEL_NAME:
    case "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    case "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

match FC_LAYER:
    case "default":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    case "arcface":
        model.fc = af.ArcFace(model.fc.in_features, num_classes)
        # model.fc = af.ArcMarginProduct(model.fc.in_features, num_classes)
    case "ccface":
        model.fc = cc.CurricularFace(model.fc.in_features, num_classes)

model.to(device)

# print(model)
summary(model, input_size=(CHANNEL_OUT, CHANNEL_IN, KERNEL_SIZE, KERNEL_SIZE))

total_batch = len(train_loader)
print("Batch count : {}".format(total_batch))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.4)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 학습 시작
train_accs, valid_accs, train_losses, valid_losses = [], [], [], []
train_acc, valid_acc, best_acc = 0, 0, 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}    [{len(train_set)}]\n-------------------------------")

    train_acc, train_loss = fit.run(device, train_loader, model, criterion, optimizer)
    valid_acc, valid_loss = valid.run(device, valid_loader, model, criterion)

    print(f"Train - Acc: {(100*train_acc):>3.2f}%, Loss: {train_loss:>10f}")
    print(f"Valid - Acc: {(100*valid_acc):>3.2f}%, Loss: {valid_loss:>10f}\n")

    train_accs.append(train_acc.cpu() * 100)
    valid_accs.append(valid_acc.cpu() * 100)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # 모델 저장
    if valid_acc > best_acc:
        torch.save(model.state_dict(), f"{WEIGHT_FILENAME}")
        print(f"Saved best state to {WEIGHT_FILENAME}\nValid acc: {best_acc:>2.5f} -> {valid_acc:>2.5f}\n")
        best_acc = valid_acc


# 학습 결과 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accs, label="train")
plt.plot(valid_accs, label="valid")
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(train_losses, label="train")
plt.plot(valid_losses, label="valid")
plt.title("Loss")

plt.legend()
plt.savefig(f"{MODEL_NAME}_{LOSS_FILENAME}.png")
plt.show()


print("Training done!")
