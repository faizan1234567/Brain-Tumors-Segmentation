import torch
import torch.nn as nn
from monai.networks.blocks.segresnet_block import ResBlock
# from research.models.UX_Net.uxnet_encoder import ux_block
from thesis.models.SegUXNet.model import SegUXNet

# # select device
device = "cuda" if torch.cuda.is_available() else "cpu"
input = torch.rand(1, 4, 128, 128, 128).to(device)
# # model = SegConvNeXt(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3).cuda()
# # model = SegConvNeXtv1(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3
# # )
model = SegUXNet(spatial_dims=3, 
                init_filters=32, 
                in_channels= 4,
                out_channels=3, 
                dropout_prob=0.2, 
                blocks_down=(1, 2, 2, 4), 
                blocks_up=(1, 1, 1), 
                enable_gc=True).to(device)
out = model(input)
print(out.shape)