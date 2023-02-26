from torchvision.io import read_image
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import os
from glob import glob


class CustomDataset(Dataset):
    def __init__(self, label, files, transform=None):
        self.files = files
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = read_image(f"{self.files[idx]}")
        label = self.label

        return image, label


def prepareCustomDataset(data_path, transform=None):
    data_sets = []

    i = 0
    for label in os.listdir(data_path):
        data_sets.append(CustomDataset(i, glob(f"{data_path}/{label}/*.jpg"), transform=transform))
        i += 1

    train_set = ConcatDataset(data_sets)

    return train_set


def getDataLoaders(train_set, test_set, batch_size_train, batch_size_test):
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True)
    valid_loader = DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)

    return train_loader, valid_loader
