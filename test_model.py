import torch

from monai.networks.nets import SegResNet

input = torch.rand(1, 2, 128, 128, 128)
out = model(input)
print(out.shape)