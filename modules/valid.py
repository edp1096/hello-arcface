import torch

def run(device, dataloader, model, loss_fn):
    model.eval()

    dataset_size = len(dataloader.dataset)
    valid_acc_top1_ratio, valid_acc_top3_ratio, valid_loss, correct_top1, correct_top3 = 0, 0, 0, 0, 0

    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)

            # 예측 오류 계산
            pred = model(image.float())
            valid_loss += loss_fn(pred, label).item() * image.size(0)

            # 정확도 계산
            correct_top1 += torch.sum(pred.argmax(1) == label.data)
            correct_top3 += torch.sum(torch.topk(pred, 3, dim=1)[1] == label.data.view(-1, 1)).item()

    valid_acc_top1_ratio = correct_top1 / dataset_size
    valid_acc_top3_ratio = correct_top3 / dataset_size
    valid_loss_ratio = valid_loss / dataset_size

    return valid_acc_top1_ratio, valid_acc_top3_ratio, valid_loss_ratio
