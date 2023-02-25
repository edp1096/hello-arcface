import torch


def runCNN(device, dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss, correct = 0, 0

    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)

            pred = model(image.float())
            valid_loss += loss_fn(pred, label).item()

            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    valid_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>.2f}%, Avg loss: {valid_loss:>8f} \n")
