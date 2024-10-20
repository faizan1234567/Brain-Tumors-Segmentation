import torch
from research.models.SegConvNeXt.segconvnext import SegConvNeXt
from research.models.SegConvNeXt.segconvnextv1 import SegConvNeXtv1
from monai.networks.nets import SegResNet
from monai.networks.nets import SwinUNETR



input = torch.rand(1, 4, 128, 128, 128).cpu()
# model = SegConvNeXt(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3).cuda()
# model = SegConvNeXtv1(spatial_dims=3, init_filters=32, in_channels=4, out_channels=3
#                   ).cpu()\
# SWIN UNETR model
model = model = SwinUNETR(
                img_size=128,
                in_channels=4,
                out_channels=3,
                feature_size=48,
                drop_rate=0.1,
                attn_drop_rate=0.2,
                dropout_path_rate=0.1,
                spatial_dims=3,
                use_checkpoint=False,
                use_v2=False).cpu()
# output
out = model(input)
print(out.shape)
