import torch
import torch.nn as nn
# credit: https://github.com/DebeshJha/ResUNetplusplus-PyTorch
class Squeeze_Excitation3D(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1, 1)
        x = inputs * x
        return x

class Stem_Block3D(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm3d(out_c),
        )

        self.attn = Squeeze_Excitation3D(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ResNet_Block3D(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm3d(in_c),
            nn.ReLU(),
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm3d(out_c),
        )

        self.attn = Squeeze_Excitation3D(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ASPP3D(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm3d(out_c)
        )

        self.c2 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm3d(out_c)
        )

        self.c3 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm3d(out_c)
        )

        self.c4 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm3d(out_c)
        )

        self.c5 = nn.Conv3d(out_c, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y

class Attention_Block3D(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm3d(in_c[0]),
            nn.ReLU(),
            nn.Conv3d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool3d((2, 2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm3d(in_c[1]),
            nn.ReLU(),
            nn.Conv3d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm3d(in_c[1]),
            nn.ReLU(),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class Decoder_Block3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block3D(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.r1 = ResNet_Block3D(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d

class ResUnetPlusPlus(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()

        self.c1 = Stem_Block3D(in_channels, 8, stride=1)
        self.c2 = ResNet_Block3D(8, 16, stride=2)
        self.c3 = ResNet_Block3D(16, 32, stride=2)
        self.c4 = ResNet_Block3D(32, 64, stride=2)

        self.b1 = ASPP3D(64, 128)

        self.d1 = Decoder_Block3D([32, 128], 64)
        self.d2 = Decoder_Block3D([16, 64], 32)
        self.d3 = Decoder_Block3D([8, 32], 16)

        self.aspp = ASPP3D(16, 8)
        self.output = nn.Conv3d(8, out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        b1 = self.b1(c4)

        d1 = self.d1(c3, b1)
        d2 = self.d2(c2, d1)
        d3 = self.d3(c1, d2)

        output = self.aspp(d3)
        output = self.output(output)

        return output



if __name__ == "__main__":
    model = ResUnetPlusPlus()
    input = torch.rand(1, 4, 32, 32, 32)
    out = model(input)
    print(out.shape)

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_res=(4, 32, 32, 32), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)