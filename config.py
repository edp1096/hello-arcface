# DATASET_NAME = "mnist" # original size=28
# DATA_ROOT = "data/mnist"

# DATASET_NAME = "stl10"  # original size=96
# DATA_ROOT = "data/stl10"

DATASET_NAME = "dst"  # original size=224
DATA_ROOT = "data/dst"

# DATASET_NAME = "facechip"  # original size=384
# DATA_ROOT = "data/facechip"

IMAGE_SIZE = 384

# MODEL_NAME = "resnet18"
MODEL_NAME = "efficientnetv2_s"

DO_VALID = False
# DO_VALID = True

USE_AMP = True
# USE_ARCFACE = False
USE_ARCFACE = True

OUTPUT_SAVE_ROOT = "weights"
COMMON_FILENAME = f"{OUTPUT_SAVE_ROOT}/{DATASET_NAME}_{MODEL_NAME}"
WEIGHT_FILE = f"{COMMON_FILENAME}.pt"
WEIGHT_INFO_FILE = f"{COMMON_FILENAME}_info.log"
SCATTER_FILE = f"{COMMON_FILENAME}_dist"
LOSS_RESULT_FILE = f"{COMMON_FILENAME}.png"


# BATCH_SIZE = 1920  # resnet18, 28
# BATCH_SIZE = 256  # resnet18, 96
# BATCH_SIZE = 128  # resnet18, 384
# BATCH_SIZE = 96  # efficientnetv2_s, 96
# BATCH_SIZE = 32  # efficientnetv2_s, 224
BATCH_SIZE = 14  # efficientnetv2_s, 384

EPOCHS_PRETRAIN = 10
EPOCHS = 40
LEARNING_RATE = 0.03
