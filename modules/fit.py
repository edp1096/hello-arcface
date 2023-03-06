from config import *

import torch
import torch.cuda.amp as amp


def run(device, loader, model, loss_fn, optimizer):
    model.train()

    dataset_size = len(loader.dataset)
    train_acc, loss_total, corrects = 0.0, 0.0, 0

    for batch, (image, label) in enumerate(loader):
        image, label = image.to(device), label.to(device)

        with torch.set_grad_enabled(True):
            # 예측 오류 계산
            # features = model(image)
            # logits = metric_fc(features, label)
            logits = model(image)
            loss = loss_fn(logits, label)
            _, pred = torch.max(logits.data, 1)

            # 역전파
            optimizer.zero_grad()
            loss.backward()

        optimizer.step()

        # 정확도 계산
        loss_total += loss.item() * image.size(0)
        corrects += (label == pred).sum()

    train_acc = corrects.item() / dataset_size
    train_loss = loss_total / dataset_size

    return train_acc, train_loss


def runAMP(device, loader, model, metric_fc, loss_fn, optimizer, optimizer_metric, scaler):
    model.train()

    dataset_size = len(loader.dataset)
    train_acc, loss_total, corrects = 0.0, 0.0, 0.0

    for batch, (image, label) in enumerate(loader):
        image, label = image.to(device), label.to(device)

        # with torch.set_grad_enabled(True) and amp.autocast():
        with amp.autocast():
            # 예측 오류 계산
            features = model(image)
            logits = metric_fc(features, label)
            loss = loss_fn(logits, label)
            _, pred = torch.max(logits.data, 1)

            # 역전파
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(optimizer_metric)
            scaler.update()

            optimizer.step()
            optimizer_metric.step()

        # 정확도 계산
        loss_total += loss.item() * image.size(0)
        corrects += (label == pred).sum()

    train_acc = corrects.item() / dataset_size
    train_loss = loss_total / dataset_size

    return train_acc, train_loss
