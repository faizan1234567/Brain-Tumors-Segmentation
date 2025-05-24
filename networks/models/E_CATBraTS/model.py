from monai.utils import set_determinism, ensure_tuple_rep
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.blocks.squeeze_and_excitation import ChannelSELayer
from monai.utils import UpsampleMode
from monai.handlers import CheckpointSaver
from monai.networks.blocks.unetr_block import  UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose
)
import numpy as np
import torch
import torch.nn as nn
import os
import datetime
from typing import Optional, Sequence, Tuple, Type, Union, Dict
try:
    from networks.models.E_CATBraTS.building_blocks import DoubleConv, ShuffleBlock
except ImportError:
    from building_blocks import DoubleConv, ShuffleBlock

class E_CATBraTS(nn.Module):
    """

    """
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        init_filters: int = 8,
        dropout_prob: Optional[float] = None,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
        num_class = 2, 
        with_BN = True, 
        channel_width = 4
    ) -> None:
        """


        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        self.act = act
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims
        )

    
     
        self.cov3d_12_en = DoubleConv(in_ch = in_channels, out_ch = 1 * feature_size, kernel_size = 3, padding = 1)
        
        self.cov3d_22_en = DoubleConv(in_ch =  feature_size, out_ch =  feature_size, kernel_size = 3, padding = 1)
        
        self.cov3d_32_en = DoubleConv(in_ch = 2 * feature_size, out_ch = 2 * feature_size, kernel_size = 3, padding = 1)
        
        self.cov3d_42_en = DoubleConv(in_ch = 4 * feature_size, out_ch = 4 * feature_size, kernel_size = 3, padding = 1)
        
        self.cov3d_4_en = DoubleConv(in_ch = 8 * feature_size, out_ch = 8 * feature_size, kernel_size = 3, padding = 1)
  
        self.cov3d_52_en = DoubleConv(in_ch = 16 * feature_size, out_ch = 16 * feature_size, kernel_size = 3, padding = 1)
  
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.shuffle1 = ShuffleBlock()
        self.shuffle2 = ShuffleBlock()
        self.shuffle3 = ShuffleBlock()
        self.shuffle4 = ShuffleBlock()
        self.shuffle5 = ShuffleBlock()

    def load_from(self, weights):

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def forward(self, x_in):
     
        
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.cov3d_12_en(x_in)
        
        enc1 = self.shuffle1(hidden_states_out[0])
        enc1 = self.cov3d_22_en(enc1)
        
        enc2 = self.shuffle2(hidden_states_out[1])
        enc2 = self.cov3d_32_en(enc2)
        
        enc3 = self.shuffle3(hidden_states_out[2])
        enc3 = self.cov3d_42_en(enc3)
        
        enc4 = self.shuffle4(hidden_states_out[3])
        enc4 = self.cov3d_4_en(enc4)
        
        dec4 = self.shuffle5(hidden_states_out[4])
        dec4 = self.cov3d_52_en(dec4)
        
        
        dec3 = self.decoder5(dec4,enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
     
        return logits
    
if __name__ == "__main__":
    set_determinism(seed=0)
    model = E_CATBraTS(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=12,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=True,
        spatial_dims=3
    )
    print(model)
    x = torch.randn(1, 4, 128, 128, 128)
    output = model(x) # this takes a fair amount of time to run
    print(output.shape)  # Expected output shape: (1, 2, 128, 128, 128)