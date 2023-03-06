from config import *
from common import device, data_transform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.manifold import TSNE

from modules.net import NetHead
from modules.file import loadWeights
import modules.plot as plotter


tsne = TSNE(random_state=0)

test_set = datasets.ImageFolder(f"{DATA_ROOT}/valid", transform=data_transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(test_set.classes)

model = NetHead(num_classes)
model.to(device)
model.load_state_dict(loadWeights()["model"])
model.eval()


# 분포도
fname = SCATTER_FILE

total_preds = torch.FloatTensor().to(device)
total_labels = torch.LongTensor().to(device)

for batch, (image, label) in enumerate(test_loader):
    image, label = image.to(device), label.to(device)

    with torch.no_grad():
        test_preds = model(image.float())
        total_preds = torch.cat((total_preds, test_preds))
        total_labels = torch.cat((total_labels, label))


with torch.no_grad():
    test_tsne_embeds = tsne.fit_transform(total_preds.cpu().detach().numpy())
    plotter.scatter(test_tsne_embeds, total_labels.cpu().numpy(), subtitle=fname, dataset=DATASET_NAME.upper())

print(f"Image file {fname}.png saved")
