import torch


def run(device, dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.view(-1, 28 * 28).to(device), y.to(device)  # linear

            pred = model(X.float())
            valid_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    valid_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>.1f}%, Avg loss: {valid_loss:>8f} \n")


def runCNN(device, dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X.float())
            valid_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    valid_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>.1f}%, Avg loss: {valid_loss:>8f} \n")


def runRNN(device, dataloader, model, loss_fn):
    model.eval()

    num_batches = len(dataloader)
    valid_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.view(-1, 28, 28).to(device) # batch, seq_dim, input_dim
            # X = X.reshape(-1, 28, 28).to(device)  # batch, seq_len, input_size

            output = model(X.float())
            _, pred = torch.max(output.data, 1)

            total += y.size(0)
            correct += (pred == y.to(device)).sum()

    valid_loss /= num_batches
    accuracy = 100 * correct / total

    # Print Loss
    print("Test error: \n Loss: {:>8f}. Accuracy: {:>.1f}".format(valid_loss, accuracy))
