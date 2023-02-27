import os
import numpy as np
from imageio import imsave

import requests
import tarfile
import matplotlib.pyplot as plt
from tqdm import tqdm


DATASET_URI = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
DOWNLOAD_PATH = "data_bins"

W, H, C = 96, 96, 3
IMG_SIZE = W * H * C

LABEL_LIST = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

DATA_ROOT = "data/stl10"
BIN_ROOT = "data_bins/stl10_binary"
BIN_PATHS = {
    "train_data": f"{BIN_ROOT}/train_X.bin",
    "train_labels": f"{BIN_ROOT}/train_y.bin",
    "test_data": f"{BIN_ROOT}/test_X.bin",
    "test_labels": f"{BIN_ROOT}/test_y.bin",
    "unlabelled_data": f"{BIN_ROOT}/unlabeled_X.bin",
}


def showSampleImage(fpath):
    f = open(fpath, "rb")
    file_size = os.path.getsize(fpath)

    num = np.random.randint(0, (file_size / IMG_SIZE) - 1)
    f.seek(IMG_SIZE * num, os.SEEK_SET)

    image = np.fromfile(f, dtype=np.uint8, count=IMG_SIZE)

    image = np.reshape(image, (C, H, W))
    image = np.transpose(image, (2, 1, 0))

    plt.title("Image #" + str(num + 1))
    plt.imshow(image)
    plt.show()


def readLABELs(path_to_labels):
    with open(path_to_labels, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def readIMGs(fpath):
    with open(fpath, "rb") as f:
        fdata = np.fromfile(f, dtype=np.uint8)

        im_data = np.reshape(fdata, (-1, C, W, H))
        im_data = np.transpose(im_data, (0, 3, 2, 1))

        return im_data


def saveIMGs(images, labels, types):
    i = 0
    for image in tqdm(images, position=0):
        dir = DATA_ROOT + "/" + types + "/" + str(LABEL_LIST[labels[i] - 1]) + "/"
        os.makedirs(dir, exist_ok=True)

        filename = dir + str(i)
        imsave("%s.png" % filename, image, format="png")

        i += 1


def saveUnlabeledIMGs(images, limit=0):
    dir = DATA_ROOT + "/" + "test" + "/"

    if len(images) < limit:
        limit = len(images)

    if limit > 0 and len(images) > limit:
        images = images[:limit]

    i = 0
    for image in tqdm(images, position=0):
        os.makedirs(dir, exist_ok=True)

        filename = dir + str(i)
        imsave("%s.png" % filename, image, format="png")

        i += 1


def download(uri, path):
    os.makedirs(path, exist_ok=True)

    fpath = path + "/stl10_binary.tar.gz"
    if os.path.exists(fpath):
        print("'stl10_binary.tar.gz' File already exists")
        return

    r = requests.get(uri, stream=True)
    total_size_in_bytes = int(r.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(fpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                progress_bar.update(len(chunk))
                f.write(chunk)

    tar = tarfile.open(path + "/stl10_binary.tar.gz", "r:gz")
    # tar.extractall(path=path)
    for member in tqdm(tar.getmembers(), total=len(tar.getmembers())):
        if member.isreg():
            tar.extract(member, path=path)
    tar.close()


def main():
    download(DATASET_URI, DOWNLOAD_PATH)

    showSampleImage(BIN_PATHS["train_data"])

    train_labels = readLABELs(BIN_PATHS["train_labels"])
    train_images = readIMGs(BIN_PATHS["train_data"])

    test_labels = readLABELs(BIN_PATHS["test_labels"])
    test_images = readIMGs(BIN_PATHS["test_data"])
    split = int(len(test_labels) / 2)

    valid_labels = test_labels[:split]
    valid_images = test_images[:split]
    test_labels = test_labels[split:]
    test_images = test_images[split:]

    saveIMGs(train_images, train_labels, "train")
    saveIMGs(valid_images, valid_labels, "valid")
    saveIMGs(test_images, test_labels, "test")

    # unlabeled_images = readIMGs(BIN_PATHS["test_data"])
    # limit = int(len(train_images) / 6 * 2)
    # saveUnlabeledIMGs(unlabeled_images, limit)


if __name__ == "__main__":
    main()
