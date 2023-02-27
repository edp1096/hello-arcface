# DATASET_NAME = "mnist"
# DATA_ROOT = "data/mnist"
# IMAGE_SIZE = 28

# DATASET_NAME = "cifar10"
# DATA_ROOT = "data/cifar10"
# IMAGE_SIZE = 32

DATASET_NAME = "stl10"
DATA_ROOT = "data/stl10"
IMAGE_SIZE = 96

"""
model: resnet50 only
fc layer: default, arcface, ccface
'default' means nn.Linear
'arcface' means modules.arcface.ArcFace
'ccface' means modules.ccface.CurricularFace
"""
MODEL_NAME = "resnet50"
FC_LAYER = "arcface"
WEIGHT_BASE_FILENAME = f"{MODEL_NAME}_{DATASET_NAME}_{FC_LAYER}_base.pt"
WEIGHT_FILENAME = f"{MODEL_NAME}_{DATASET_NAME}_{FC_LAYER}.pt"
SCATTER_FILENAME = f"dist_{DATASET_NAME}_{FC_LAYER}"
LOSS_FILENAME = f"loss_{DATASET_NAME}_{FC_LAYER}"

EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.03

CHANNEL_IN = 3  # INPUT CHANNEL
CHANNEL_OUT = 64  # OUTPUT CHANNEL
KERNEL_SIZE = 7  # KERNEL SIZE
