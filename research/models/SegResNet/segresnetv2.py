from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from utils import conv_layer
from experimental import deformableResBlock

class SegResNetv2(nn.Module):
    """
    SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module does not include the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        in_channels: int = 4,
        out_channels: int = 3,
        dropout_prob: float | None = 0.05,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1, 1),
        dims=[32, 64, 128, 256],
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        self.dims = dims
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})

        self.downsample_layers = nn.ModuleList() 
        self.stem_layer = nn.Sequential(
            conv_layer(spatial_dims, init_filters//2 , init_filters, kernel_size=(4, 4, 4), stride=(4, 4, 4),
                           dropout=dropout_prob, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=init_filters),
        )
        self.downsample_layers.append(self.stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout_prob, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)
        
        self.norm = norm
        self.init_conv = get_conv_layer(spatial_dims=3, in_channels=4, out_channels=init_filters//2, kernel_size= 3, 
                                        )
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.down_layers = self._make_down_layers()
        self.up_layers = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)
        

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        for i, item in enumerate(self.blocks_down):
            down_layer = nn.Sequential(
                self.downsample_layers[i], *[deformableResBlock(spatial_dims=3, in_channels=self.dims[i]) for _ in range(item)] 
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers = nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = int(filters * 2 ** (n_up - (i + 1)))
            up_layers.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode, scale_factor= 2 if i < 3 else 4),
                        *[deformableResBlock(spatial_dims=3, in_channels= sample_in_channels//2) for _ in range(blocks_up[i])]
                    ]
                )
            )
        return up_layers
    
    # Final Convolution
    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters//2),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters//2, out_channels, kernel_size=1, bias=True),
        )
    # Encoder 
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.init_conv(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []
        down_x.append(x)
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x
    
    # Decoder
    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, upl in enumerate(self.up_layers):
            x = upl(x) + down_x[i + 1]

        if self.use_conv_final:
            x = self.conv_final(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse() 

        x = self.decode(x, down_x)
        return x