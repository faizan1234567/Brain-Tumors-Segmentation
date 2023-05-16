

import numpy as np


import torch
import torch.nn as nn
from torch.nn import Conv3d, BatchNorm3d, ReLU, MaxPool3d
from torch.functional import F


class DoubleConv(nn.Module):
    "(Conv3d-->BatchNorm3D-->ReLU) * 2"
    def __init__(self, in_channels, out_channels, mid_channels = None, kernel_size = 3, padding =1, is_bais = False) -> None:
        super().__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.is_bais = is_bais
        self.mid_channels = mid_channels
        if self.mid_channels is None:
            self.mid_channels = self.out_channels
        

        self.double_conv = nn.Sequential( 
                                    Conv3d(in_channels= self.in_channels,
                                           out_channels= self.mid_channels,
                                           kernel_size= self.kernel_size,
                                           padding= self.padding,
                                           bias= self.is_bais),
                                    BatchNorm3d(self.mid_channels),
                                    ReLU(inplace = True),

                                    Conv3d(in_channels= self.mid_channels,
                                           out_channels= self.out_channels,
                                           kernel_size= self.kernel_size,
                                           padding= self.padding,
                                           bias= self.is_bais),
                                    BatchNorm3d(self.out_channels),
                                    ReLU(inplace= True))
    def forward(self, x):
        return self.double_conv(x)
    
    class Down(nn.module):
        """(max-pool --> dobule convolution)"""
        def __init__(self, in_channels, 
                     out_channels, 
                     middle_channels = None, 
                     kernel_size=3,
                     padding = 1,
                     is_bais = False,
                     pooling_kernel = 2):
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.middle_channels = middle_channels
            self.padding = padding
            self.kernel_size = kernel_size
            self.is_bias = is_bais
            self.pooling_kernel = pooling_kernel
            if self.middle_channels is None:
                self.middle_channels = self.out_channels
            self.down_sample = nn.Sequential(
                                        MaxPool3d(pooling_kernel),
                                        DoubleConv(self.in_channels, 
                                                   self.out_channels,
                                                   self.middle_channels,
                                                   self.kernel_size,
                                                   self.padding,
                                                   self.is_bias))
        def forward(self, x):
            x = self.down_sample(x)
            return x
    
class Up(nn.module):
    """concate and apply up convolution"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 middle_channels = None,
                 padding = 1,
                 kernel_size = 3,
                 is_bias = False,
                 bilinear = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.middle_channels = middle_channels
        self.padding = padding
        self.is_bias = is_bias
        self.kernel_size = kernel_size
        if self.middle_channels is None:
            self.middle_channels = out_channels
        
        self.bilinear = bilinear

        if self.bilinear:
            self.up = nn.Upsample(scale_factor= 2, mode = 'bilinear', align_corners= True)
            self.double_conv = DoubleConv(self.in_channels,
                                          self.out_channels, 
                                          self.in_channels//2,
                                          self.kernel_size,
                                          self.padding,
                                          self.is_bias)
        else:
            self.up = nn.ConvTranspose3d(self.in_channels, self.in_channels // 2, 
                                         kernel_size=2, stride=2)
            
            self.double_conv = DoubleConv(self.in_channels,
                                          self.out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x
    

            

    







if __name__ == "__main__":
    input = torch.rand(4, 155, 240, 240)
    input = input[None, ...]
    input = input.to(torch.float32)
    double_conv = DoubleConv(4, 8, 6)
    output = double_conv(input)
    print(output.shape)