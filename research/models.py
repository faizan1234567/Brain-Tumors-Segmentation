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
    