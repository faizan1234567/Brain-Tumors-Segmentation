from convolutions import convolution
from buillding_block import ResidualBlock
from models import SegResNet
import torch


input = torch.rand(4, 155, 240, 240, dtype = torch.float32, requires_grad = True)
input = input[None, :, :, :, :] #adding batch dim

# model = convolution(spatial_dims=3, in_channels= 4, out_channels=16, kernel_size= (3,3,3),
#                     strides= (1, 1, 1), dropout= 0.3, conv_only= True)

# model = ResidualBlock(spatial_dims=3, in_channels=4, kernel_size=3)
model = SegResNet(3, 4, 4, 3, 0.3)

(output, down_out) = model.encode(input)
print(output.shape)
print(len(down_out))
# print(output.dtype)