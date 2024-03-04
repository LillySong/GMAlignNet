import torch
import torch.nn as nn
import math
from .sync_batchnorm import SynchronizedBatchNorm3d
from AM import AMV2 as align
class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation
    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class Ghost_3x3x1(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(Ghost_3x3x1, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv3d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm3d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_channels, new_channels, (3,3,1), 1, (1,1,0), groups=init_channels, bias=False),
            nn.BatchNorm3d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class Conv_1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x




class Conv_3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_down(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_down, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class RHDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(RHDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = Ghost_3x3x1(self.out_inter_dim, self.out_inter_dim)
        self.conv_3x3x1_2 = Ghost_3x3x1(self.out_inter_dim, self.out_inter_dim)
        self.conv_3x3x1_3 = Ghost_3x3x1(self.out_inter_dim, self.out_inter_dim)
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        # 第一个1x1x1卷积
        x_1 = self.conv_1x1x1_1(x)
        # 四分
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        # 经过3x3x1卷积
        x2_ = self.conv_3x3x1_1(x2)
        x3_ = self.conv_3x3x1_2(x2_ + x3)
        x4_ = self.conv_3x3x1_3(x3_ + x4)
        # 结合残差
        x2 = x2 + x2_
        x3 = x3 + x3_
        x4 = x4 + x4_
        # 拼接
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x3x3(x + x_1)
        return x_1
class HDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(HDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        # 三个 3x3x1卷积
        self.conv_3x3x1_1 = Ghost_3x3x1(self.out_inter_dim, self.out_inter_dim)
        self.conv_3x3x1_2 = Ghost_3x3x1(self.out_inter_dim, self.out_inter_dim)
        self.conv_3x3x1_3 = Ghost_3x3x1(self.out_inter_dim, self.out_inter_dim)
        # 两个 1x1x1卷积
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        # 一个 1x1x3卷积
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        # 为啥所有通道数是32，因为每个通道数是8
        x_1 = self.conv_1x1x1_1(x)
        #----分成四个通道
        x1 = x_1[:, 0:self.out_inter_dim, ...] # [1, 8, 64, 64, 64]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]

        # x2 = C(x2) 经过3x3x1卷积
        x2 = self.conv_3x3x1_1(x2)
        # x3 = C(x2+x3)
        x3 = self.conv_3x3x1_2(x2 + x3)
        # x4 = C(x3+x4)
        x4 = self.conv_3x3x1_3(x3 + x4)
        # 将四个融合
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        # 经过1x1x1卷积
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x3x3(x + x_1)
        return x_1


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        SynchronizedBatchNorm3d(out_dim),
        activation)


device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.device("cuda")

# 周期混洗操作
def hdc(image, num=2):
    x1 = torch.Tensor([]).to(device1)
    for i in range(num):
        for j in range(num):
            for k in range(num):
                x3 = image[:, :, k::num, i::num, j::num]
                x1 = torch.cat((x1, x3), dim=1)
                # print("hdc",x1.shape)
                # # 32 x 64
    return x1


class GMAlignNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters=32):
        super(GMAlignNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_f = num_filters
        self.activation = nn.ReLU(inplace=False)

        # down
        self.conv_3x3x3 = Conv_3x3x3(self.n_f, self.n_f, self.activation)
        self.conv_1 = RHDC_module(self.n_f, self.n_f, self.activation)
        self.down_1 = Conv_down(self.n_f, self.n_f, self.activation)

        self.conv_2 = RHDC_module(self.n_f, self.n_f, self.activation)
        self.down_2 = Conv_down(self.n_f, self.n_f, self.activation)

        self.conv_3 = RHDC_module(self.n_f, self.n_f, self.activation)
        self.down_3 = Conv_down(self.n_f, self.n_f, self.activation)

        # bridge
        self.bridge = RHDC_module(self.n_f, self.n_f, self.activation)

        # align model
        self.align = align.AlignedModule(inplane_h=num_filters, inplane_l=num_filters, outplane=num_filters)

        # up
        self.up_1 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_4 = HDC_module(self.n_f * 2, self.n_f, self.activation)

        self.up_2 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_5 = HDC_module(self.n_f * 2, self.n_f, self.activation)

        self.up_3 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_6 = HDC_module(self.n_f * 2, self.n_f, self.activation)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.out = nn.Conv3d(self.n_f, out_dim, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encode
        x1 = hdc(x)  # 32x64

        x2 = self.conv_3x3x3(x1)
        x2 = self.conv_1(x2)
        x2 = self.down_1(x2)  # 32x32

        x3 = self.conv_2(x2)
        x3 = self.down_2(x3)  # 32x16

        x4 = self.conv_3(x3)
        x4 = self.down_3(x4)  # 32x8
        x4 = self.bridge(x4)  # 32x8

        # align model
        a_y1 = self.align(x3, x4)  # 32x16
        # Decode
        y1 = self.up_1(x4)  # 32x16
        y1 = y1 + a_y1
        y1 = torch.cat((y1, x3), dim=1)  # 64x16
        y1 = self.conv_4(y1)  # 32x16

        a_y2 = self.align(x2, y1)
        y2 = self.up_2(y1)  # 32x16
        y2 = y2 + a_y2
        y2 = torch.cat((y2, x2), dim=1)  # 64x32
        y2 = self.conv_5(y2)  # 32x32

        a_y3 = self.align(x1, y2)
        y3 = self.up_3(y2)  # 32x64
        y3 = y3 + a_y3
        y3 = torch.cat((y3, x1), dim=1)
        y3 = self.conv_6(y3)  # 32x64

        y = self.upsample(y3)  # 32x128
        y = self.out(y)  # 4x128
        out = self.softmax(y)

        return out

