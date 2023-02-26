import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt


im = cv2.imread("Lenna.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

xfrm = transforms.ToTensor()
im_t = xfrm(im)

print(im_t.shape)

plt.imshow(im_t.permute(1, 2, 0))
plt.show()
