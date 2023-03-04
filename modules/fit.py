import torch
import torch.cuda.amp as amp


def run(device, loader, model, loss_fn, optimizer):
    model.train()

    dataset_size = len(loader.dataset)
    train_acc_ratio, train_loss, corrects = 0.0, 0.0, 0

    for batch, (image, label) in enumerate(loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # 예측 오류 계산
            pred = model(image.float())
            loss = loss_fn(pred, label)

            # 역전파
            loss.backward()
            optimizer.step()

        # 정확도 계산
        train_loss += loss.item() * image.size(0)
        corrects += torch.sum(pred.argmax(1) == label.data)

    train_acc_ratio = corrects / dataset_size
    train_loss_ratio = train_loss / dataset_size

    return train_acc_ratio, train_loss_ratio


def runAMP(device, loader, model, loss_fn, optimizer, scaler):
    model.train()

    dataset_size = len(loader.dataset)
    train_acc_ratio, train_loss, corrects = 0.0, 0.0, 0

    for batch, (image, label) in enumerate(loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        with amp.autocast() and torch.set_grad_enabled(True):
            # 예측 오류 계산
            pred = model(image)
            loss = loss_fn(pred, label)

            # 역전파
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # 정확도 계산
        train_loss += loss.item() * image.size(0)
        corrects += torch.sum(pred.argmax(1) == label.data)

    train_acc_ratio = corrects / dataset_size
    train_loss_ratio = train_loss / dataset_size

    return train_acc_ratio, train_loss_ratio
