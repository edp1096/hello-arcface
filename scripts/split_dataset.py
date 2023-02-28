import os
import shutil


SRC_ROOT = "data/src"
DST_ROOT = "data/dst"

DST_ROOT_TRAIN = f"{DST_ROOT}/train"
DST_ROOT_VALID = f"{DST_ROOT}/valid"
DST_ROOT_TEST = f"{DST_ROOT}/test"

os.makedirs(DST_ROOT, exist_ok=True)
os.makedirs(DST_ROOT_TRAIN, exist_ok=True)
os.makedirs(DST_ROOT_VALID, exist_ok=True)
# os.makedirs(DST_ROOT_TEST, exist_ok=True)

for dir in os.listdir(SRC_ROOT):
    src_root = os.path.join(SRC_ROOT, dir)
    dst_train = os.path.join(DST_ROOT_TRAIN, dir)
    dst_valid = os.path.join(DST_ROOT_VALID, dir)

    if os.path.isdir(src_root) == False:
        continue

    print(dir)

    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_valid, exist_ok=True)

    i = 0
    for file in os.listdir(src_root):
        dst_path = dst_train
        if i >= 150:
            dst_path = dst_valid

        file_src = os.path.join(src_root, file)
        file_dst = os.path.join(dst_path, file)

        shutil.copyfile(file_src, file_dst)

        i += 1
