""" 
https://tutorials.pytorch.kr/beginner/basics/buildmodel_tutorial.html
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, channel=1, in_features=28 * 28, out_features=10):
        super().__init__()

        availavle_vram = int(torch.cuda.get_device_properties(0).total_memory * 0.8 * 0.000001)

        # mid_features = 512
        mid_features = int(in_features * 0.65)
        if mid_features * out_features * 4 > availavle_vram:
            mid_features = int(availavle_vram)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, out_features),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)

        return x


# device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)


tbatch, tchan, theight, twidth = 1, 3, 28, 28
im_shape = (tbatch, tchan, twidth, theight)

input_features = tchan * twidth * theight
num_classes = 10

model = NeuralNetwork(channel=tchan, in_features=input_features, out_features=num_classes).to(device)
print(model)


loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

inputs = []
sample_num = 20
for i in range(sample_num):
    random_input = torch.rand(im_shape, device=device)
    random_label = torch.randint(0, num_classes, (1,), device=device)

    print(f"input: {random_input.shape}, label: {random_label.item()}")

    inputs.append({"input": random_input, "label": random_label})


epochs_num = 20
for i in range(epochs_num):
    model.train()
    for tdata in inputs:
        logit = model(tdata["input"])
        loss = loss_fn(logit, tdata["label"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch: {i+1:>3}, loss: {loss.item():>.3f}")


model.eval()
for tdata in inputs:
    logit = model(tdata["input"])

    pred = logit.argmax(1)

    print(f"actual: {tdata['label'].item():>2}, pred: {pred.item():>1}")


model.eval()
one_input = torch.rand(im_shape, device=device)
logit = model(one_input)
pred = logit.argmax(1)

print(f"pred: {pred.item():>1}")
