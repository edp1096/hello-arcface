""" 
https://tutorials.pytorch.kr/beginner/basics/buildmodel_tutorial.html
"""

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, in_features=28 * 28, out_features=10):
        super().__init__()

        self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer(x)
        x = self.fc(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)


tbatch, tchan, twidth, theight = 1, 3, 28, 28
im_shape = (tbatch, tchan, twidth, theight)

input_features = tchan * twidth * theight
num_classes = 10

model = NeuralNetwork(in_features=input_features, out_features=num_classes).to(device)
print(model)


loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

inputs = []
for i in range(10):
    random_input = torch.rand(im_shape, device=device)
    random_label = torch.randint(0, num_classes, (1,), device=device)

    inputs.append({"input": random_input, "label": random_label})


model.train()
for i in range(20):
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

    pred_probab = nn.Softmax(dim=1)(logit)
    pred = pred_probab.argmax(1)

    print(f"actual: {tdata['label'].item():>2}, pred: {pred.item():>1}")
