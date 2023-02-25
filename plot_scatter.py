# # 2D
# import matplotlib.pyplot as plt

# feat2d = [[0.0, 0], [1.0, 1], [2.0, 2], [3.0, 3], [4.0, 4], [5.0, 5], [6.0, 6], [7.0, 7], [8.0, 8], [9.0, 9]]

# f = plt.figure(figsize=(16, 9))
# c = ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#990000", "#999900", "#009900", "#009999"]
# for i in range(10):
#     plt.plot(feat2d[i][0], feat2d[i][1], ".", c=c[i])
# plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

# plt.grid()
# plt.show()

# plt.pause()


# Other 2D
import torch
from torchvision.datasets import MNIST
from torchvision import datasets, models, transforms
from sklearn.manifold import TSNE

import modules.dataset as dset
import modules.plot as plotter

tsne = TSNE(random_state=0)

batch_size = 512

""" My dataset """
test_transform = transforms.ToTensor()
test_data = dset.prepareCustomDataset(9, "datas/test", test_transform)

batch_image = torch.zeros((500, 28, 28))
batch_label = torch.zeros(500)

classes_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
j=0
for i in range(test_data.__len__()):
    image, label = test_data[i]

    if classes_count[label] > 50:
        continue

    batch_image[j] = image
    batch_label[j] = label
    j += 1

    classes_count[label] += 1

batch_image = batch_image.float()
batch_label = batch_label.float()

print(batch_image.shape)
print(batch_label.shape)

batch_x_2d = tsne.fit_transform(batch_image.flatten(1))
plotter.scatter(batch_x_2d, batch_label.cpu().numpy(), subtitle="My MNIST test set")


""" Built-in dataset """
test_data = MNIST("./MNIST/", train=False, download=True)

(vis_test_batch_x, vis_test_batch_y) = (test_data.data[:batch_size], test_data.targets[:batch_size])
vis_test_batch_x = vis_test_batch_x.float()
vis_test_batch_y = vis_test_batch_y.float()

print(vis_test_batch_x.shape)
print(vis_test_batch_y.shape)

vis_test_batch_x_2d = tsne.fit_transform(vis_test_batch_x.flatten(1))
plotter.scatter(vis_test_batch_x_2d, vis_test_batch_y.cpu().numpy(), subtitle="Original MNIST test set")
