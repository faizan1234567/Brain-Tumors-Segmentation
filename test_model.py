import torch
from research.models.SegConvNeXt.segconvnext import SegConvNeXt


input = torch.rand(1, 4, 160, 192, 128).cuda()
model = SegConvNeXt(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3).cuda()

# output
out = model(input)
print(out.shape)
