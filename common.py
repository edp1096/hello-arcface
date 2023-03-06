from config import *

from torchvision import transforms

import modules.xfrm as xfrm


def getTransformSet():
    data_transform = transforms.Compose(
        [
            xfrm.CLAHE(clipLimit=2.5),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return data_transform


