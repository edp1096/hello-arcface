import torch
from torchvision.io import read_image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, ConcatDataset

from glob import glob


class MNIST(Dataset):
    def __init__(self, label, files, transform=None):
        self.files = files
        self.label = label
        self.transform = transform

        # print(f"MNIST: {label} {len(files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = read_image(f"{self.files[idx]}")
        # label = torch.tensor([self.label])
        label = self.label

        return image, label


def prepareCustomDataset(last_num: int, data_path, transform=None):
    data_files = []
    for i in range(0, last_num):
        data_files.append(glob(f"{data_path}/{i}/*.jpg"))

    data_sets = []
    for i in range(0, last_num):
        data_sets.append(MNIST(i, data_files[i], transform=transform))

    train_set = ConcatDataset(data_sets)

    return train_set


def prepareTorchvisionDataset(train_transform=None, valid_transform=None):
    train_set = datasets.MNIST(root="data/MNIST_data/", train=True, transform=train_transform, download=True)
    valid_set = datasets.MNIST(root="data/MNIST_data/", train=False, transform=valid_transform, download=True)

    return train_set, valid_set


def getDataLoaders(train_set, test_set, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)

    return train_loader, valid_loader
