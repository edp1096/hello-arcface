from config import *

import torch
from torchvision import transforms

import modules.xfrm as xfrm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


data_transform = transforms.Compose(
    [
        xfrm.CLAHE(clipLimit=2.5),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
