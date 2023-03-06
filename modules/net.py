from common import device
from config import *

import torch.nn as nn
from torchvision import datasets, models, transforms
# from facenet_pytorch import MTCNN, InceptionResnetV1
from modules.inception_resnet_v1 import InceptionResnetV1

from modules.loss import ArcFace


class NetHead(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(NetHead, self).__init__()

        in_features = None
        weights = None

        match MODEL_NAME:
            case "resnet50":
                if pretrained:
                    weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.model = models.resnet50(weights=weights)
                in_features, self.model.fc = self.model.fc.in_features, nn.Identity()
            case "mobilenetv3":
                if pretrained:
                    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                self.model = models.mobilenet_v3_small(weights=weights)
                in_features, self.model.classifier[3] = self.model.classifier[3].in_features, nn.Identity()
            case "efficientnetv2_s":
                if pretrained:
                    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
                self.model = models.efficientnet_v2_s(weights=weights)
                in_features, self.model.classifier[1] = self.model.classifier[1].in_features, nn.Identity()
            case "inceptionresnetv1":
                if pretrained:
                    weights = "vggface2"
                    # weights = "casia-webface"
                self.model = InceptionResnetV1(pretrained=weights, num_classes=num_classes, device=device)
                in_features = self.model.last_linear.out_features

        self.fc = nn.Linear(in_features, num_classes)
        self.arcface = ArcFace(in_features, num_classes, s=30.0, m_arc=0.5)

    def forward(self, x, label=None):
        x = self.model(x)
        x = self.fc(x)

        if label is not None:
            x = self.arcface(x, label)

        return x
