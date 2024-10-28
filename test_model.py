import torch
import torch.nn as nn
from monai.networks.blocks.segresnet_block import ResBlock
from research.models.UX_Net.uxnet_encoder import ux_block
from research.models.SegUXNet.model import SegUXNet

device = "cuda" if torch.cuda.is_available() else "cpu"
input = torch.rand(1, 4, 128, 128, 128).to(device)
# model = SegConvNeXt(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3).cuda()
# model = SegConvNeXtv1(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3
#                   ).cpu()



model = SegUXNet(spatial_dims=3, init_filters=32, 
                 in_channels=4, out_channels=3, dropout_prob=0.2, 
                 )
out = model(input)
print(out.shape)