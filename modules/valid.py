import torch


def run(device, dataloader, model, loss_fn):
    model.eval()

    dataset_size = len(dataloader.dataset)
    valid_acc, loss_total, corrects = 0.0, 0.0, 0.0

    for image, label in dataloader:
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            # 예측 오류 계산
            # features = model(image)
            # logits = metric_fc(features, label)
            logits = model(image)
            loss = loss_fn(logits, label)

            _, pred = torch.max(logits.data, 1)

        # 정확도 계산
        corrects += (pred == label.data).sum()
        loss_total += loss.item() * image.size(0)

    valid_acc = corrects.item() / dataset_size
    valid_loss = loss_total / dataset_size

    return valid_acc, valid_loss
