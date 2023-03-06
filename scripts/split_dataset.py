import os
import shutil
import glob
import math

SRC_ROOT = "data/src"
DST_ROOT = "data/dst"
# SRC_ROOT = "raw100"
# DST_ROOT = "vggface-my100"

DST_ROOT_TRAIN = f"{DST_ROOT}/train"
DST_ROOT_VALID = f"{DST_ROOT}/valid"
DST_ROOT_TEST = f"{DST_ROOT}/test"

os.makedirs(DST_ROOT, exist_ok=True)
os.makedirs(DST_ROOT_TRAIN, exist_ok=True)
os.makedirs(DST_ROOT_VALID, exist_ok=True)
os.makedirs(DST_ROOT_TEST, exist_ok=True)

extTYPEs = ["*.jpg", "*.png", "*.jpeg", "*.bmp"]

limit_total, limit_train, limit_valid = 180, 150, 20
limit_total_orig, limit_train_orig, limit_valid_orig = limit_total, limit_train, limit_valid


"""
Required images >= "limit_total"
* train: "limit_train"
* valid: "limit_valid"
* test: else of them
"""

LACK_LOG_CREATED = False
cnt = 0
for dir in os.listdir(SRC_ROOT):
    limit_total, limit_train, limit_valid = limit_total_orig, limit_train_orig, limit_valid_orig

    target_dir_name = dir.replace("new_", "").replace("_6point", "").lower()

    src_dir = os.path.join(SRC_ROOT, dir)
    dst_train = os.path.join(DST_ROOT_TRAIN, target_dir_name)
    dst_valid = os.path.join(DST_ROOT_VALID, target_dir_name)
    dst_test = os.path.join(DST_ROOT_TEST, target_dir_name)

    if os.path.isdir(src_dir) == False:
        continue

    print(f"{dir:>34s} -> {target_dir_name:>34s}", end=" ")

    image_files = []
    for ext in extTYPEs:
        image_files += glob.glob(os.path.join(src_dir, ext))

    if len(image_files) < limit_total:
        with open(f"{DST_ROOT}/not_enough_images.log", "a") as f:
            f.write(f"{dir} {len(image_files)}\n")

        LACK_LOG_CREATED = True
        print(" /  LACK")
        continue

    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_valid, exist_ok=True)
    os.makedirs(dst_test, exist_ok=True)

    limit_total = min(limit_total, len(image_files))
    if limit_total != limit_total_orig:
        limit_train = min(limit_train, int(math.ceil(limit_total * 0.8)))
        limit_valid = min(limit_valid, int(math.ceil(limit_total * 0.16)))

    print(f"({len(image_files)} {limit_total} {limit_train} {limit_valid}) ", end=" ")

    i, j, k = 0, 0, 0
    # for file in os.listdir(src_root):
    for file in image_files:
        if i > limit_total:
            break

        fname = os.path.basename(file)
        fext = os.path.splitext(fname)[1]

        # target_fname = fname
        target_fname = f"{i}.{fext}"

        dst_path = dst_train
        if i + j + k > limit_total:
            print(f"({i} {j} {k}) ", end=" ")
            break
        elif j >= limit_valid:
            dst_path = dst_test
            target_fname = f"{k}.{fext}"
            k += 1
        elif i >= limit_train:
            dst_path = dst_valid
            target_fname = f"{j}.{fext}"
            j += 1
        else:
            i += 1

        file_src = file
        file_dst = os.path.join(dst_path, target_fname)
        shutil.copyfile(file_src, file_dst)

    cnt += 1
    print(" /  OK")

print(f"Total {cnt} directories processed")
