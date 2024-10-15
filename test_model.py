import torch
from research.models.SegConvNeXt.segconvnext import SegConvNeXt
from research.models.SegConvNeXt.segconvnextv1 import SegConvNeXtv1
from monai.networks.nets import SegResNet



input = torch.rand(1, 4, 128, 128, 128).cpu()
# model = SegConvNeXt(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3).cuda()
model = SegConvNeXtv1(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3
                  ).cpu()

# output
out = model(input)
print(out.shape)
