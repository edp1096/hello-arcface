from config import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from lion_pytorch import Lion

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

match FC_LAYER:
    case "default":
        model.classifier[1] = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(model.classifier[1].in_features, num_classes))

    case "arcface":
        model.fc = nn.Sequential(nn.Dropout(p=0.4, inplace=True), af.ArcFace(model.fc.in_features, num_classes, s=64.0, m=0.5))
        # model.fc = nn.Sequential(nn.Dropout(p=0.4, inplace=True), af.ArcFace(model.fc.in_features, num_classes))

    case "ccface":
        model.fc = nn.Sequential(nn.Dropout(p=0.4, inplace=True), cc.CurricularFace(model.fc.in_features, num_classes))

model.to(device)
# model.load_state_dict(torch.load(WEIGHT_BASE_FILENAME))

# print(model)
summary(model, input_size=(CHANNEL_OUT, CHANNEL_IN, KERNEL_SIZE, KERNEL_SIZE))

total_batch = len(train_loader)
print("Batch count : {}".format(total_batch))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = Lion(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4)


# 학습 시작
train_accs, valid_accs_top1, valid_accs_top3, train_losses, valid_losses = [], [], [], [], []
train_acc, valid_acc_top1, valid_acc_top3, best_acc = 0, 0, 0, 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}    [{len(train_set)}]\n-------------------------------")

    train_acc, train_loss = fit.run(device, train_loader, model, criterion, optimizer)
    valid_acc_top1, valid_acc_top3, valid_loss = valid.run(device, valid_loader, model, criterion)

    print(f"Train - Acc: {(100*train_acc):>3.2f}%, Loss: {train_loss:>10f}")
    print(f"Valid - Acc top1: {(100*valid_acc_top1):>3.2f}%, top3: {(100*valid_acc_top3):>3.2f}%, Loss: {valid_loss:>10f}")

    train_accs.append(train_acc.cpu() * 100)
    valid_accs_top1.append(valid_acc_top1.cpu() * 100)
    valid_accs_top3.append(valid_acc_top3 * 100)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # 모델 저장
    if valid_acc_top1 > best_acc:
        torch.save(model.state_dict(), f"{WEIGHT_FILENAME}")
        print(f"Saved best state to {WEIGHT_FILENAME}\nValid acc: {best_acc:>2.5f} -> {valid_acc_top1:>2.5f}\n")
        best_acc = valid_acc_top1


# 학습 결과 그래프
plt.figure(figsize=(12, 4))

plt.title("Accuracy")
plt.subplot(1, 2, 1)
plt.plot(train_accs, label="train")
plt.plot(valid_accs_top1, label="valid top1")

plt.title("Loss")
plt.subplot(1, 2, 2)
plt.plot(train_losses, label="train")
plt.plot(valid_losses, label="valid")

plt.legend()
plt.savefig(f"{MODEL_NAME}_{LOSS_FILENAME}.png")
plt.show()


print("Training done!")
