import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.io import ImageReadMode
from torchvision.transforms import ToTensor
from sklearn.manifold import TSNE

import modules.dataset as dset
import modules.network as net
import modules.arcface as af
import modules.plot as plotter


# model_fname = "model_mnist_default.pt"
model_fname = "model_mnist_arcface.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# random.seed(777)
# torch.manual_seed(777)
# if device == "cuda":
#     torch.cuda.manual_seed_all(777)

tsne = TSNE(random_state=0)

num_classes = 10

test_transform = transforms.Compose([transforms.ToTensor()])
test_data = dset.prepareCustomDataset(9, "datas/test", test_transform)

# resnet
model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
# model.fc = nn.Linear(512, 10) # resnet18
# model.fc = nn.Linear(2048, 10)  # resnet50
num_ftrs = model.fc.in_features
model.fc = af.ArcFace(num_ftrs, num_classes)

model.to(device)
model.load_state_dict(torch.load(model_fname))
model.eval()


# size_limit = 500
size_limit = test_data.__len__()

batch_image = torch.zeros((size_limit, 28, 28))
batch_label = torch.zeros(size_limit)

classes_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
j = 0
for i in range(test_data.__len__()):
    image, label = test_data[i]

    if classes_count[label] > int(size_limit / 10):
        continue

    batch_image[j] = image
    batch_label[j] = label
    j += 1

    classes_count[label] += 1

batch_image = batch_image.float()
batch_label = batch_label.float()

with torch.no_grad():
    test_outputs = model(batch_image.unsqueeze(1).to(device))  # cnn
    test_tsne_embeds = tsne.fit_transform(test_outputs.cpu().detach().numpy())

# fname = "distribution_linear_test"
fname = "distribution_arcface_test"
plotter.scatter(test_tsne_embeds, batch_label.cpu().numpy(), subtitle=fname)

print("Done")
