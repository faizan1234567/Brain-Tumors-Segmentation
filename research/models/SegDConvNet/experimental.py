import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer
from dcn.modules.deform_conv import *


class deformableResBlock(nn.Module):
    """
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: Union[Tuple, str]= ("GROUP", {"num_groups": 4}),
        kernel_size: int = 3,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.dconv1 = DeformConvPack_d(in_channels=in_channels, out_channels=in_channels, 
                                       kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), dimension="HW")
        self.dconv2 = DeformConvPack_d(in_channels=in_channels, out_channels=in_channels, 
                                       kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), dimension="HW")

    def forward(self, x):

        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.dconv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.dconv2(x)

        x += identity

        return x
