# DATASET_NAME = "mnist"
# DATA_ROOT = "data/mnist"
# IMAGE_SIZE = 28

DATASET_NAME = "stl10"
DATA_ROOT = "data/stl10"
IMAGE_SIZE = 96

# DATASET_NAME = "vggface-my100"
# DATA_ROOT = "data/vggface-my100"
# IMAGE_SIZE = 224

# DATASET_NAME = "dst"
# DATA_ROOT = "data/dst"
# IMAGE_SIZE = 224


# MODEL_NAME = "resnet50"
# MODEL_NAME = "mobilenetv3"
MODEL_NAME = "efficientnetv2_s"

USE_AMP = False
# USE_AMP = True

COMMON_FILENAME = f"{DATASET_NAME}_{MODEL_NAME}"
WEIGHT_BASE_FILENAME = f"{COMMON_FILENAME}_base.pt"
WEIGHT_FILENAME = f"{COMMON_FILENAME}.pt"
SCATTER_FILENAME = f"{COMMON_FILENAME}_dist"
LOSS_FILENAME = f"{COMMON_FILENAME}"


EPOCHS = 20
# BATCH_SIZE = 1920  # resnet50, 28
# BATCH_SIZE = 256  # resnet50, 96
BATCH_SIZE = 96  # efficientnetv2, 96
# BATCH_SIZE = 32  # efficientnetv2, 224
LEARNING_RATE = 0.1
