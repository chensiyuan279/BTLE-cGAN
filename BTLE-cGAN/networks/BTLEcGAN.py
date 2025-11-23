import torch.nn as nn
import torch
from thop import profile
from torchsummary import summary
from torch.cuda.amp import autocast
import gc
from networks.modules.MSDConv_SSFC import MSDConv_SSFC


class First_DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(First_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.Conv = nn.Sequential(
            MSDConv_SSFC(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            MSDConv_SSFC(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.Conv(input)


import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicKernelModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DynamicKernelModule, self).__init__()
        self.kernel_size = kernel_size
        # 3x3卷积用于生成掩膜特征图F_m^{dk}
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # 1x1卷积用于生成动态卷积核
        self.conv1x1_kernel = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 1x1卷积用于生成最终的精细化掩膜
        self.conv1x1_final = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, Fdk_m, Fdk_o, estimated_mask):
        # 1. 生成掩膜特征图 Fdk_m，通过3x3卷积
        Fdk_m = self.conv3x3(Fdk_m)

        # 2. 将特征图 Fdk_o 与估计掩膜 M^ 逐元素相乘，得到 masked_feature
        masked_feature = Fdk_o * estimated_mask

        # 3. 对 masked_feature 进行池化，大小变为 3x3
        pooled_feature = F.adaptive_avg_pool2d(masked_feature, (3, 3))

        # 4. 通过 1x1 卷积生成动态卷积核
        dynamic_kernel = self.conv1x1_kernel(pooled_feature)

        # 5. 确保 dynamic_kernel 的形状正确，调整为与输入通道数一致的深度卷积核
        batch_size, channels, _, _ = dynamic_kernel.shape
        dynamic_kernel = dynamic_kernel.view(batch_size * channels, 1, self.kernel_size, self.kernel_size)

        # 6. 使用深度卷积，将动态卷积核应用于 Fdk_o，注意groups参数设置为32
        Fdk_o_reshaped = Fdk_o.view(1, batch_size * channels, Fdk_o.size(2), Fdk_o.size(3))
        residual_feature = F.conv2d(Fdk_o_reshaped, dynamic_kernel, stride=1, padding=1, groups=batch_size * channels)
        residual_feature = residual_feature.view(batch_size, channels, Fdk_o.size(2), Fdk_o.size(3))

        # 7. 将残差特征图 residual_feature 与 Fdk_m 相加，得到精细化的特征图 Fdk_m~
        refined_feature_map = Fdk_m + residual_feature

        # 8. 使用 1x1 卷积和 Sigmoid 激活生成精细化掩膜 M~
        refined_mask = torch.sigmoid(self.conv1x1_final(refined_feature_map))

        return refined_mask, self.conv1x1_final(refined_feature_map)


class _NonLocalBlock2D_EGaussian(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlock2D_EGaussian, self).__init__()

        assert dimension in (1, 2, 3)

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):  # 例: 1, 128, 32, 32
        batch_size = x.size(0)  # 1
        # 128, 32, 32--64, 32, 32--64, 16, 16--1, 64, 16*16
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # 1, 16*16, 64
        # print(f'g_x:{g_x.shape}')

        # 128, 32, 32--64, 32, 32--1, 64, 32*32
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # 1, 32*32, 64
        # print(f'theta_x:{theta_x.shape}')

        # 128, 32, 32--64, 32, 32--64, 16, 16--1, 64, 16*16
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)  # 1, 32*32, 16*16
        # print(f'f:{f.shape}')
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)  # 1, 32*32, 64
        y = y.permute(0, 2, 1).contiguous()  # 1, 64, 32*32
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # 1, 64, 32, 32
        # print(f'y:{y.shape}')
        W_y = self.W(y)  # 1, 128, 32, 32
        z = W_y + x  # 1, 128, 32, 32

        return z

class NONLocalBlock2D_EGaussian(_NonLocalBlock2D_EGaussian):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D_EGaussian, self).__init__(in_channels,
                                                        inter_channels=inter_channels,
                                                        dimension=2, sub_sample=sub_sample,
                                                        bn_layer=bn_layer)



class BTLEcGAN(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=0.5):
        super(BTLEcGAN, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = First_DoubleConv(in_ch, int(64 * ratio))
        self.Conv1_2 = First_DoubleConv(in_ch, int(64 * ratio))
        self.Conv2_1 = DoubleConv(int(64 * ratio), int(128 * ratio))
        self.Conv2_2 = DoubleConv(int(64 * ratio), int(128 * ratio))
        self.Conv3_1 = DoubleConv(int(128 * ratio), int(256 * ratio))
        self.Conv3_2 = DoubleConv(int(128 * ratio), int(256 * ratio))
        self.Conv4_1 = DoubleConv(int(256 * ratio), int(512 * ratio))
        self.Conv4_2 = DoubleConv(int(256 * ratio), int(512 * ratio))
        self.Conv5_1 = DoubleConv(int(512 * ratio), int(1024 * ratio))
        self.Conv5_2 = DoubleConv(int(512 * ratio), int(1024 * ratio))

        self.Up5 = nn.ConvTranspose2d(int(1024 * ratio), int(512 * ratio), 2, stride=2)
        self.Up_conv5 = DoubleConv(int(1024 * ratio), int(512 * ratio))

        self.Up4 = nn.ConvTranspose2d(int(512 * ratio), int(256 * ratio), 2, stride=2)
        self.Up_conv4 = DoubleConv(int(512 * ratio), int(256 * ratio))

        self.Up3 = nn.ConvTranspose2d(int(256 * ratio), int(128 * ratio), 2, stride=2)
        self.Up_conv3 = DoubleConv(int(256 * ratio), int(128 * ratio))

        self.Up2 = nn.ConvTranspose2d(int(128 * ratio), int(64 * ratio), 2, stride=2)
        self.Up_conv2 = DoubleConv(int(128 * ratio), int(64 * ratio))

        self.Up_conv22 = DoubleConv(int(64 * ratio), int(64 * ratio))

        self.Conv_1x1 = nn.Conv2d(int(64 * ratio), out_ch, kernel_size=1, stride=1, padding=0)

        self.DKModule = DynamicKernelModule(int(64 * ratio), 1, kernel_size=3)

        # self.nonlocal1 = NONLocalBlock2D_EGaussian(int(64 * ratio))
        # self.nonlocal2 = NONLocalBlock2D_EGaussian(int(128 * ratio))
        self.nonlocal3 = NONLocalBlock2D_EGaussian(int(256 * ratio))
        self.nonlocal4 = NONLocalBlock2D_EGaussian(int(512 * ratio))
        self.nonlocal5 = NONLocalBlock2D_EGaussian(int(1024 * ratio))


    def forward(self, x1, x2):
        # encoding
        # x1, x2 = torch.unsqueeze(x1[0], dim=0), torch.unsqueeze(x1[1], dim=0)
        c1_1 = self.Conv1_1(x1)
        c1_2 = self.Conv1_2(x2)
        x1 = torch.abs(torch.sub(c1_1, c1_2))
        # x11 = self.nonlocal1(x1)

        c2_1 = self.Maxpool(c1_1)
        c2_1 = self.Conv2_1(c2_1)
        c2_2 = self.Maxpool(c1_2)
        c2_2 = self.Conv2_2(c2_2)
        x2 = torch.abs(torch.sub(c2_1, c2_2))
        # x22 = self.nonlocal2(x2)

        c3_1 = self.Maxpool(c2_1)
        c3_1 = self.Conv3_1(c3_1)
        c3_2 = self.Maxpool(c2_2)
        c3_2 = self.Conv3_2(c3_2)
        x3 = torch.abs(torch.sub(c3_1, c3_2))
        x33 = self.nonlocal3(x3)

        c4_1 = self.Maxpool(c3_1)
        c4_1 = self.Conv4_1(c4_1)
        c4_2 = self.Maxpool(c3_2)
        c4_2 = self.Conv4_2(c4_2)
        x4 = torch.abs(torch.sub(c4_1, c4_2))
        x44 = self.nonlocal4(x4)

        c5_1 = self.Maxpool(c4_1)
        c5_1 = self.Conv5_1(c5_1)
        c5_2 = self.Maxpool(c4_2)
        c5_2 = self.Conv5_2(c5_2)
        x5 = torch.abs(torch.sub(c5_1, c5_2))
        x55 = self.nonlocal5(x5)

        # decoding
        d5 = self.Up5(x55)
        d5 = torch.cat((x44, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x33, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d22 = self.Up_conv22(d2)
        # print(d22.shape)
        # print(d2.shape)   ##torch.Size([16, 32, 256, 256])
        d1 = self.Conv_1x1(d22)
        out = nn.Sigmoid()(d1)
        # print(out.shape)   ##torch.Size([16, 1, 256, 256])
        refined_mask, t2 = self.DKModule(d22, d2, out)  # 将倒数第二层特征图 d2 和 估计的掩膜 out 传入 DKM
        # print(t2.shape)
        return refined_mask, t2, d1  # 返回精细化后的掩膜

        # return refined_mask  # 返回精细化后的掩膜

        # return out


if __name__ == "__main__":
    A2016 = torch.randn(1, 3, 256, 256)
    A2019 = torch.randn(1, 3, 256, 256)
    model = BTLEcGAN(3, 1, ratio=0.5)
    out_result = model(A2016, A2019)
    summary(model, [(3, 256, 256), (3, 256, 256)])
    flops, params = profile(model, inputs=(A2016, A2019))
    print(flops, params)
