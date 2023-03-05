# DATASET_NAME = "mnist"
# DATA_ROOT = "data/mnist"
# IMAGE_SIZE = 28

# DATASET_NAME = "stl10"
# DATA_ROOT = "data/stl10"
# IMAGE_SIZE = 96

# DATASET_NAME = "vggface-my100"
# DATA_ROOT = "data/vggface-my100"
# IMAGE_SIZE = 224

DATASET_NAME = "dst"
DATA_ROOT = "data/dst"
IMAGE_SIZE = 224

"""
model: resnet50, effnetv2s
fc layer: default, myloss
'default' default nn.Linear
'myloss' means modules.myloss.MyLoss
"""
MODEL_NAME = "resnet50"
FC_LAYER = "myloss"

USE_AMP = True

COMMON_FILENAME = f"{DATASET_NAME}_{MODEL_NAME}_{FC_LAYER}"
WEIGHT_BASE_FILENAME = f"{COMMON_FILENAME}_base.pt"
WEIGHT_FILENAME = f"{COMMON_FILENAME}.pt"
SCATTER_FILENAME = f"{COMMON_FILENAME}_dist"
LOSS_FILENAME = f"{COMMON_FILENAME}"


# resnet50, 8GB, 28
# EPOCHS = 20
# BATCH_SIZE = 1920
# LEARNING_RATE = 0.03

# resnet50, 8GB, 96
# EPOCHS = 20
# BATCH_SIZE = 192
# LEARNING_RATE = 0.03

# effnetv2s, 8GB, 96
# EPOCHS = 20
# BATCH_SIZE = 96
# LEARNING_RATE = 0.03

# resnet50, 8GB, 224
EPOCHS = 20
BATCH_SIZE = 18
LEARNING_RATE = 0.03

# effnetv2s, 8GB, 224
# EPOCHS = 20
# BATCH_SIZE = 18
# LEARNING_RATE = 0.03


CHANNEL_IN = 3  # INPUT CHANNEL
CHANNEL_OUT = 64  # OUTPUT CHANNEL
KERNEL_SIZE = 7  # KERNEL SIZE
