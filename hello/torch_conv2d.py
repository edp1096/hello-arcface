import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tbatch, tchan, theight, twidth = 1, 3, 28, 28
im_shape = (tbatch, tchan, twidth, theight)

one_input = torch.rand(im_shape, device=device)

ochan = tchan * 3
conv2d = torch.nn.Conv2d(tchan, ochan, kernel_size=3).to(device)
print(conv2d.stride[0], conv2d.stride[1])
print(conv2d.dilation[0], conv2d.dilation[1])

maxpool2d = torch.nn.MaxPool2d(tchan, ochan).to(device)
print(maxpool2d.stride)
print(maxpool2d.dilation)
print(maxpool2d.kernel_size)
print(maxpool2d.padding)

# hout = (theight + 2 * conv2d.padding[0] - conv2d.dilation[0] * (conv2d.kernel_size[0] - 1) - 1) / conv2d.stride[0] + 1
# wout = (twidth + 2 * conv2d.padding[1] - conv2d.dilation[1] * (conv2d.kernel_size[1] - 1) - 1) / conv2d.stride[1] + 1
hout = (theight + 2 * maxpool2d.padding - maxpool2d.dilation * (maxpool2d.kernel_size - 1) - 1) / maxpool2d.stride + 1
wout = (twidth + 2 * maxpool2d.padding - maxpool2d.dilation * (maxpool2d.kernel_size - 1) - 1) / maxpool2d.stride + 1
print(hout, wout)

print(tchan * 3 * int(hout) * int(wout))

flatten = torch.nn.Flatten()
layer = torch.nn.Sequential(
    torch.nn.Linear(ochan * int(hout) * int(wout), 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
).to(device)

out = conv2d(one_input)
print("conv2d:", out.shape[1], out.shape[2], out.shape[3])

out = maxpool2d(out)
print("maxpool2d:", out.shape[1], out.shape[2], out.shape[3])

out = flatten(out)
print(out.shape)

out = layer(out)
print(out.shape)
