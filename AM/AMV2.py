import torch
import torch.nn as nn
import torch.nn.functional as F

class Point_Reinforcement(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Point_Reinforcement, self).__init__()
        self.conv_atten = nn.Conv3d(in_chan, in_chan, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv3d(in_chan, out_chan, kernel_size=1, bias=False)
    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool3d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat
class AlignedModule(nn.Module):
    def __init__(self, inplane_h, inplane_l, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv3d(inplane_h, outplane, 1, bias=False)
        self.down_l = nn.Conv3d(inplane_l, outplane, 1, bias=False)
        self.flow_make = nn.Conv3d(outplane * 2, 3, kernel_size=kernel_size, padding=1, bias=False)
        self.att = Point_Reinforcement(in_chan=inplane_l, out_chan=inplane_l)

    def forward(self, x,y):
        low_feature = x
        h_feature = y
        h_feature_orign = h_feature
        d, h, w = low_feature.size()[2:]
        size = (d, h, w)

        low_feature = self.down_l(low_feature)
        low_feature = self.att(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="trilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):

        out_d, out_h, out_w = size #
        n, c, d, h, w = input.size() # 高级语义信息

        w = torch.arange(0, out_w).view(1, 1, -1).repeat(out_d, out_h, 1).view(1, out_d, out_h, out_w) # [1, 32, 32, 32]
        h = torch.arange(0, out_h).view(1, -1, 1).repeat(out_d, 1, out_w).view(1, out_d, out_h, out_w)
        d = torch.arange(0, out_d).view(-1, 1, 1).repeat(1, out_h, out_w).view(1, out_d, out_h, out_w)


        grid = torch.cat((w, h, d), 0).float() # [3, 32, 32, 32]

        grid = grid.repeat(n, 1, 1, 1, 1).type_as(input).to(input.device)  # 只用这里的grid的话，就是线性插值 [1, 3, 32, 32, 32]

        vgrid = grid + flow #[1, 3, 32, 32, 32]

        norm = torch.tensor([[[[[out_w, out_h, out_d]]]]]).type_as(input).to(input.device) # [1, 1, 1, 1, 3]

        vgrid = vgrid.permute(0, 2, 3, 4, 1) / norm # [1, 32, 32, 32, 3]

        output = F.grid_sample(input, vgrid)
        return output
