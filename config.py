# DATASET_NAME = "mnist"
# DATA_ROOT = "data/mnist"
# IMAGE_SIZE = 28

DATASET_NAME = "stl10"
DATA_ROOT = "data/stl10"
IMAGE_SIZE = 96

# DATASET_NAME = "dst"
# DATA_ROOT = "data/dst"
# IMAGE_SIZE = 224


# MODEL_NAME = "resnet50"
# MODEL_NAME = "mobilenetv3"
MODEL_NAME = "efficientnetv2_s"
# MODEL_NAME = "inceptionresnetv1"

# USE_AMP = False
USE_AMP = True

OUTPUT_SAVE_ROOT = "weights"
COMMON_FILENAME = f"{OUTPUT_SAVE_ROOT}/{DATASET_NAME}_{MODEL_NAME}"
WEIGHT_FILE = f"{COMMON_FILENAME}.pt"
WEIGHT_INFO_FILE = f"{COMMON_FILENAME}_info.log"
SCATTER_FILE = f"{COMMON_FILENAME}_dist"
LOSS_RESULT_FILE = f"{COMMON_FILENAME}.png"


# BATCH_SIZE = 1920  # resnet50, 28
# BATCH_SIZE = 256  # resnet50, 96
BATCH_SIZE = 96  # efficientnetv2, 96
# BATCH_SIZE = 32  # efficientnetv2, 224
EPOCHS = 40
LEARNING_RATE = 0.03
