import torch
from building_blocks import ConvNextBlock2D, ConvNextBlock3D
from segresnetv1 import SegResNetv1
from segresnetv2 import SegResNetv2
from utils import conv_layer
from experimental import deformableResBlock

if __name__ == "__main__":
    x = torch.rand(1, 4, 128, 128, 128).cuda()
 
    model = SegResNetv2(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3).cuda()

    
    
    out = model(x)
    print(out.shape)