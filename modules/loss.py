import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import linalg


class MyA(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, input):
        weight_norm = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        input_norm = linalg.norm(input, dim=1, keepdim=True)

        output = self.fc(input) / (input_norm * weight_norm)

        return output


class My(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        output = F.linear(F.normalize(input), F.normalize(self.weight))

        return output
