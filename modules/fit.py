def runCNN(device, loader, model, loss_fn, optimizer):
    model.train()

    size = len(loader.dataset)

    for batch, (image, label) in enumerate(loader):
        image, label = image.to(device), label.to(device)

        # 예측 오류 계산
        pred = model(image.float())
        loss = loss_fn(pred, label)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(image)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
