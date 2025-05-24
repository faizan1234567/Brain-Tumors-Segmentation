

# Shuffle Block and DoubleConv with Channel Attention
import torch
import torch.nn as nn
from monai.networks.blocks.squeeze_and_excitation import ChannelSELayer

class ShuffleBlock(nn.Module):
    def __init__(self, groups=4):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
       
        b, N, C, H, W = x.size()
        g = self.groups
        return x.view(b, N, g, C//g, H, W).permute(0, 2, 1, 3, 4,5).reshape(b,N, C, H, W)

# Double convolution block with channel attention    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size = 3, stride = 1,
                 padding = 1, dilation = 1, groups = 1, bias = True):
        super(DoubleConv, self).__init__()
        
        
        self.se_layer = ChannelSELayer(
            spatial_dims=3, in_channels=out_ch, r=2, acti_type_1="relu", acti_type_2="sigmoid"
        )
        
        self.relu=nn.LeakyReLU(inplace=True)
    
        self.cov = nn.Conv3d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel_size,
                             stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias)
        self.bn = nn.BatchNorm3d(num_features = out_ch)
       

    def forward(self, input):
        
        return self.relu(self.se_layer(  self.bn(self.cov(input))))