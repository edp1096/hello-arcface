from config import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.manifold import TSNE

import modules.arcface as af
import modules.ccface as cc
import modules.plot as plotter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tsne = TSNE(random_state=0)

# test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = transforms.ToTensor()
test_set = datasets.ImageFolder(f"{DATA_ROOT}/test", transform=test_transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(test_set.classes)

model = models.resnet18()
match FC_LAYER:
    case "default":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    case "arcface":
        model.fc = af.ArcFace(model.fc.in_features, num_classes)
        # model.fc = af.ArcMarginProduct(model.fc.in_features, num_classes)
    case "ccface":
        model.fc = cc.CurricularFace(model.fc.in_features, num_classes)

model.to(device)
model.load_state_dict(torch.load(WEIGHT_FILENAME))
model.eval()


# 분포도
fname = f"{SCATTER_FILENAME}_test"

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
