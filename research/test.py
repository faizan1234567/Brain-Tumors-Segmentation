from convolutions import convolution
from buillding_block import ResidualBlock
from models import SegResNet
import torch
import time



input = torch.rand(4, 155, 240, 240, dtype = torch.float32, requires_grad = True)
input = input[None, :, :, :, :] #adding batch dim

# model = convolution(spatial_dims=3, in_channels= 4, out_channels=16, kernel_size= (3,3,3),
#                     strides= (1, 1, 1), dropout= 0.3, conv_only= True)

model = ResidualBlock(spatial_dims=3, in_channels=4, kernel_size=3)
total_params = sum(
	param.numel() for param in model.parameters()
)

tic = time.time()
output = model(input)
toc = time.time()
duration = toc  - tic
# print(total_params)
print(duration * 1000)
# model = SegResNet(spatial_dims=3, init_kernels=8, in_channels=4, out_channels=16, 
#                   dropout_prob=0.3, num_groups=4, blocks_down= (1, 2, 2, 4), blocks_up= (1, 1, 1))

# (output, shapes) = model(input)
print(output.shape)
# print(shapes)
# print([1].shape/)
# print(output.dtype)