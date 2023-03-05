from config import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.cuda.amp as amp

import modules.fit as fit
import modules.valid as valid
import modules.loss as myloss
import modules.xfrm as xfrm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

scaler = None
if USE_AMP:
    BATCH_SIZE *= 2
    scaler = amp.GradScaler()

# data_transform = transforms.ToTensor()
data_transform = transforms.Compose(
    [
        xfrm.CLAHE(clipLimit=2.5),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)
train_set = datasets.ImageFolder(f"{DATA_ROOT}/train", transform=data_transform)
valid_set = datasets.ImageFolder(f"{DATA_ROOT}/valid", transform=data_transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(train_set.classes)


# # Show one image in dataset
# random_idx = torch.randint(0, len(train_set), (1,)).item()
# tdata = train_set[random_idx][0]
# tlabel = train_set[random_idx][1]
# print(f"{random_idx}/{len(train_set)}", tdata.shape)

# ndata = tdata.permute(1, 2, 0).numpy()
# nlabel = train_set.classes[tlabel]

# resize_ndata = cv2.resize(ndata, (512, 512), interpolation=cv2.INTER_AREA)
# cv2.imshow(nlabel, resize_ndata)
# cv2.waitKey(0)


match MODEL_NAME:
    case "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    case "effnetv2s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

match FC_LAYER:
    case "myloss":
        match MODEL_NAME:
            case "resnet50":
                model.fc = nn.Sequential(nn.Dropout(p=0.4, inplace=True), myloss.My(model.fc.in_features, num_classes))
            case "effnetv2s":
                model.classifier[1] = nn.Sequential(nn.Dropout(p=0.4, inplace=True), myloss.My(model.classifier[1].in_features, num_classes))

model.to(device)
# model.load_state_dict(torch.load(WEIGHT_BASE_FILENAME))

# print(model)
summary(model, input_size=(CHANNEL_OUT, CHANNEL_IN, KERNEL_SIZE, KERNEL_SIZE))

total_batch = len(train_loader)
print("Batch count : {}".format(total_batch))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


# 학습 시작
total_start_time = time.time()
train_accs, valid_accs_top1, valid_accs_top3, train_losses, valid_losses = [], [], [], [], []
train_acc, valid_acc_top1, valid_acc_top3, best_acc = 0, 0, 0, 0
best_epoch = 0
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}    [{len(train_set)}]\n-------------------------------")

    epoch_start_time = time.time()

    if USE_AMP:
        train_acc, train_loss = fit.runAMP(device, train_loader, model, criterion, optimizer, scaler)
    else:
        train_acc, train_loss = fit.run(device, train_loader, model, criterion, optimizer)

    valid_acc_top1, valid_acc_top3, valid_loss = valid.run(device, valid_loader, model, criterion)

    pad = " "
    print(f"Train - Acc:      {(100*train_acc):>3.2f}%, {pad:<12} Loss: {train_loss:>3.5f}")
    print(f"Valid - Acc: top1 {(100*valid_acc_top1):>3.2f}%, top3 {(100*valid_acc_top3):>3.2f}%, Loss: {valid_loss:>3.5f}")

    train_accs.append(train_acc.cpu() * 100)
    valid_accs_top1.append(valid_acc_top1.cpu() * 100)
    valid_accs_top3.append(valid_acc_top3 * 100)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f"Epoch time: {time.time() - epoch_start_time:.2f} seconds\n")

    # 모델 저장
    if valid_acc_top1 > best_acc:
        torch.save(model.state_dict(), f"{WEIGHT_FILENAME}")
        print(f"Saved best model state to {WEIGHT_FILENAME}")
        torch.save(optimizer.state_dict(), f"{WEIGHT_FILENAME}o")
        print(f"Saved best optimizer state to {WEIGHT_FILENAME}o")
        if USE_AMP:
            torch.save(scaler.state_dict(), f"{WEIGHT_FILENAME}a")
            print(f"Saved best scaler state to {WEIGHT_FILENAME}a")

        print(f"Valid acc: {best_acc:>2.5f} -> {valid_acc_top1:>2.5f}\n")

        # Write to text file
        with open(f"{WEIGHT_FILENAME}.txt", "w") as f:
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Train acc: {train_acc * 100:>2.5f}%\n")
            f.write(f"Valid acc top1: {valid_acc_top1 * 100:>2.5f}%\n")
            f.write(f"Valid acc top3: {valid_acc_top3 * 100:>2.5f}%\n")
            f.write(f"Train loss: {train_loss:>2.5f}\n")
            f.write(f"Valid loss: {valid_loss:>2.5f}\n")

        best_epoch = epoch
        best_acc = valid_acc_top1

print(f"Total time: {time.time() - total_start_time:.2f} seconds")
print(f"Best epoch: {best_epoch+1}, Best acc: {best_acc.cpu() * 100:>2.5f}%")


# 학습 결과 그래프
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f"{DATASET_NAME} - {MODEL_NAME}/{FC_LAYER}")

ax[0].set_title("Accuracy")
ax[0].plot(train_accs, label="train")
ax[0].plot(valid_accs_top1, label="valid top1")
ax[0].plot(valid_accs_top3, label="valid top3")
ax[0].plot([best_epoch + 1], [best_acc.cpu() * 100], marker="o", markersize=5, color="red", label="best")
ax[0].set_ylim(0, 100)
ax[0].legend()

ax[1].set_title("Loss")
ax[1].plot(train_losses, label="train")
ax[1].plot(valid_losses, label="valid")
ax[1].legend()

plt.savefig(f"{LOSS_FILENAME}.png")
plt.show()

print("Training done!")
