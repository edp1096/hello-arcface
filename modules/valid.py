import torch

def run(device, dataloader, model, loss_fn):
    model.eval()

    dataset_size = len(dataloader.dataset)
    valid_acc_ratio, valid_loss, correct = 0, 0, 0

    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)

            # 예측 오류 계산
            pred = model(image.float())
            valid_loss += loss_fn(pred, label).item() * image.size(0)

            # 정확도 계산
            correct += torch.sum(pred.argmax(1) == label.data)

    valid_acc_ratio = correct / dataset_size
    valid_loss_ratio = valid_loss / dataset_size

    return valid_acc_ratio, valid_loss_ratio
