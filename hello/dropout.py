import torch
import torch.nn as nn

# Dropout only y

torch.manual_seed(1)
x = torch.randn(4, 4)
print("x:", x)

dropout = nn.Dropout(p=0.3, inplace=False)
y = dropout(x)

print("x:", x)
print("y:", y)


# Dropout both x and y

torch.manual_seed(1)
x = torch.randn(4, 4)
print("x:", x)

dropout = nn.Dropout(p=0.3, inplace=True)
y = dropout(x)

print("x:", x)
print("y:", y)

