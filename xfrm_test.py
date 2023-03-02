from config import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import modules.xfrm as xfrm

import cv2
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


vanilla_transform = transforms.Compose(
    [
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)
xfrm_transform = transforms.Compose(
    [
        xfrm.CLAHE(clipLimit=2.5),
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),
    ]
)
vanilla_set = datasets.ImageFolder(f"{DATA_ROOT}/valid", transform=vanilla_transform)
xfrmed_set = datasets.ImageFolder(f"{DATA_ROOT}/valid", transform=xfrm_transform)


# Show one image in dataset
random_idx = torch.randint(0, len(vanilla_set), (1,)).item()

tdata_vanilla = vanilla_set[random_idx][0]
tdata_xfrm = xfrmed_set[random_idx][0]

tlabel = vanilla_set[random_idx][1]
print(f"{random_idx}/{len(vanilla_set)}", tdata_vanilla.shape)

ndata_vanilla = tdata_vanilla.permute(1, 2, 0).numpy()
ndata_xfrm = tdata_xfrm.permute(1, 2, 0).numpy()

nlabel = vanilla_set.classes[tlabel] + " - vanilla / xfrm"

resize_vanilla = cv2.resize(ndata_vanilla, (512, 512), interpolation=cv2.INTER_AREA)
resize_xfrm = cv2.resize(ndata_xfrm, (512, 512), interpolation=cv2.INTER_AREA)


# cv2.imshow(nlabel, cv2.cvtColor(resize_vanilla, cv2.COLOR_RGB2BGR))
concat = np.concatenate((resize_vanilla, resize_xfrm), axis=1)
cv2.imshow(nlabel, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
