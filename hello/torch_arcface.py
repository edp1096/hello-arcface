import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import math


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m_arc=0.50, m_am=0.0):
        super(ArcFace, self).__init__()

        self.s = s
        self.m_arc = m_arc
        self.m_am = m_am

        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.cos_margin = math.cos(m_arc)
        self.sin_margin = math.sin(m_arc)
        self.min_cos_theta = math.cos(math.pi - m_arc)

    def forward(self, embbedings, label):
        embbedings = F.normalize(embbedings, dim=1)
        kernel_norm = F.normalize(self.weight, dim=0)
        # cos_theta = torch.mm(embbedings, kernel_norm).clamp(-1, 1)
        cos_theta = torch.mm(embbedings, kernel_norm).clip(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin

        # torch.where doesn't support fp16 input
        is_half = cos_theta.dtype == torch.float16

        cos_theta_m = torch.where(cos_theta > self.min_cos_theta, cos_theta_m, cos_theta.float() - self.m_am)
        if is_half:
            cos_theta_m = cos_theta_m.half()

        output = cos_theta * 1.0
        if label is not None:
            index = torch.zeros_like(cos_theta)
            index.scatter_(1, label.data.view(-1, 1), 1)
            index = index.byte().bool()
            output[index] = cos_theta_m[index]
            output *= self.s

        return output


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp

        return loss.mean()


class NeuralNetwork(nn.Module):
    def __init__(self, channel=1, in_features=28 * 28, out_features=10):
        super().__init__()

        availavle_vram = int(torch.cuda.get_device_properties(0).total_memory * 0.8 * 0.000001)

        mid_features = int(in_features * 0.65)
        if mid_features * out_features * 4 > availavle_vram:
            mid_features = int(availavle_vram)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, mid_features)
        self.arcface = ArcFace(mid_features, out_features, s=30.0, m_arc=0.25, m_am=0.0)

    def forward(self, x, label=None):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.arcface(x, label)

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


# loss_fn = nn.CrossEntropyLoss().to(device)
loss_fn = FocalLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

inputs = []
sample_num = 20
# sample_num = 40
# sample_num = 180
# sample_num = 1400
for i in range(sample_num):
    random_input = torch.rand(im_shape, device=device)
    random_label = torch.randint(0, num_classes, (1,), device=device)

    # print(f"input: {random_input.shape}, label: {random_label.item()}")
    print(f"input: {random_input.shape}, label: {random_label}")

    inputs.append({"input": random_input, "label": random_label})


# epochs_num = 120
# epochs_num = 30
epochs_num = 2000
model.train()
fin_loss_count = 0
for i in range(epochs_num):
    for batch, data in enumerate(inputs):
        logit = model(data["input"], data["label"])
        loss = loss_fn(logit, data["label"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch: {i+1:>3}, loss: {loss.item():>.5f}")
    if loss.item() < 0.001:
        fin_loss_count += 1
    if fin_loss_count > 3:
        break


model.eval()
for batch, data in enumerate(inputs):
    logit = model(data["input"])

    pred = logit.argmax(1)

    print(f"actual: {data['label'].item():>2}, pred: {pred.item():>1}")


model.eval()
one_input = torch.rand(im_shape, device=device)
logit = model(one_input)
pred = logit.argmax(1)

print(f"pred: {pred.item():>1}")
