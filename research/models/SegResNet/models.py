"""implementation of deep learning models here
credit: MONAI"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from buillding_block import ResidualBlock

class SegResNet(nn.Module):
    """Auto encoder decoder structured proposed by <https://arxiv.org/pdf/1810.11654.pdf>
    The model suppports 2D or 3D input 
    
    Args:
        spatial_dims: spatial dimension of the input data default value is 3
        init_filters: initial value of filters, filter size of the first convolution
        default value of init_filters is 8
        in_channels: input channels of the network, default to 4
        out_channels: output channels of the network, defalut value is 2
        dropout_prob: dropout probablity, to randomly switch of the weights, defalut to None
        act: activation type and arguments, default value is ReLU
        norm: feature normalization type defautl value is GROUP
        num_groups: number of groups in the normalization function
        use_conv_final: whether to add final convoltion layer to the output, default to True
        blocks_down: number of down sample blocks in the encoder, default ot [1, 2, 2, 4]
        blocks_up: number of up sample blocks in each layers, dafault to [1, 1, 1]
        
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - deconv, uses transposed convolution layers.
            - nontrainable, uses non-trainable `linear` interpolation.
            - pixelshuffle, uses :py:class:`monai.networks.blocks.SubpixelUpsample"""
    
    def __init__(self, 
                 spatial_dims: int = 3,
                 init_kernels: int = 8,
                 in_channels: int =  4,
                 out_channels: int = 3,
                 dropout_prob: Optional[float] = 0.3,
                 act: Union[Tuple, str] = ("RELU", {"inplace": True}),
                 norm: Union[Tuple, str] = ("GROUP", {"num_groups": 4}),
                 norm_name: str = "",
                 num_groups: int = 4,
                 use_conv_final: bool = True,
                 blocks_down: tuple = (1, 2, 2, 4),
                 blocks_up: tuple = (1, 1, 1),
                 upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE) -> None:

                 super().__init__()
                 if spatial_dims not in (2, 3):
                    raise ValueError("spatial dimension should either 2 or 3.")

                 self.spatial_dims = spatial_dims
                 self.init_filters = init_kernels
                 self.in_channels = in_channels
                 self.out_channels = out_channels
                 self.dropout_prob = dropout_prob
                 self.activation = act
                 self.activation_mode = get_act_layer(self.activation)
                 self.blocks_down = blocks_down
                 self.blocks_up = blocks_up

                 if norm_name:
                    if norm_name.lower() != "group":
                        raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
                    norm = ("group", {"num_groups": num_groups})

                 self.norm = norm
                 self.upsample_mode = UpsampleMode(upsample_mode)
                 self.use_conv_final = use_conv_final
                 self.initial_conv = get_conv_layer(spatial_dims, in_channels, init_kernels, 
                                                    kernel_size = 3, stride = 1, bias = True)

                 self.down_layers = self._make_down_layers()
                 self.up_layers, self.up_samples = self._make_up_layers()
                 self.conv_final = self._make_final_conv(out_channels)

                 if dropout_prob is not None:
                    self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)
               
   
    def _make_down_layers(self):
        """create down layers for the encoder part, in the module the input spatial size will reduce
        while the feature size will increase to learn more rich features
        """
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i # filter * 1; filter * 2; filter * 4; filter * 8; filter * 16 .... filter * 2**(n)
            pre_conv = (get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2) if i > 0 else nn.Identity())
            down_layer = nn.Sequential(pre_conv, *[ResidualBlock(spatial_dims, layer_in_channels, norm) for j in range(item)])
            down_layers.append(down_layer)
        return down_layers
    
    def _make_up_layers(self):
        """make up layers for increasing the spatial size and decreasing the feature size"""
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm)

        n_up = len(blocks_up)
        for i in range(n_up):
            #if filter size is 4 and n_up is 3 we will reduce feature size by 2 as follows (explanation):
            #filter * 2 **(3 -0) --> filter * 8 --> 8 * 8 -->64
            #filter * 2 **(3 -1) --> filter * 4 --> 8 * 4 -->32
            #filter * 2 **(3 -2) --> filter * 2 --> 8 * 2 -->16 
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResidualBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.activation)
                        for _ in range(blocks_up[i])
                    ]))

            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]))
        
        return up_layers, up_samples
    
    #add: final convolution layer 
    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.activation_mode)

    #encoder layers
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # print("Before intial convolution: {}".format(x.shape))
        x = self.initial_conv(x)
        # print("After intial convolution: {}".format(x.shape))
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    #decoder layers
    def decode(self, x: torch.Tensor, down_x: List[torch.Tensor]) -> Tuple[torch.Tensor, list]:
        shapes = []
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            shapes.append(x.shape)
            shapes.append(down_x[i + 1].shape)
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)

        return x,shapes


    #features update  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()
        print(x.shape, type(down_x))
        x,shapes = self.decode(x, down_x)
        return x, shapes
