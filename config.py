# DATASET_NAME = "mnist"
# DATA_ROOT = "data/mnist"
# IMAGE_SIZE = 28

DATASET_NAME = "cifar10"
DATA_ROOT = "data/cifar10"
IMAGE_SIZE = 32

""" 
model: ResNet18 only
fc layer: default, arcface, ccface
'default' means nn.Linear
'ccface' means modules.ccface.CurricularFace
"""
MODEL_NAME = "resnet18"
FC_LAYER = "arcface"
WEIGHT_FILENAME = f"{MODEL_NAME}_{DATASET_NAME}_{FC_LAYER}.pt"
SCATTER_FILENAME = f"dist_{DATASET_NAME}_{FC_LAYER}"

EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.05

CHANNEL_IN = 3  # INPUT CHANNEL
CHANNEL_OUT = 64  # OUTPUT CHANNEL
KERNEL_SIZE = 7  # KERNEL SIZE
