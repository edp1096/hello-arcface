import torchvision.transforms as transforms

from PIL import Image


tbatch, tchan, theight, twidth = 1, 3, 224, 224

xfrm = transforms.Compose([transforms.Resize((theight, twidth)), transforms.ToTensor()])

IMG_FILENAME = "leopard.jpg"

img = Image.open(IMG_FILENAME)
img = xfrm(img)

print(img.shape)
