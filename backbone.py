import torch
import torch.nn as nn
import math

# DarknetConv2D_BN_Leaky模块
class DarknetConv2D_BN_Leaky(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size, stride=1):
        super(DarknetConv2D_BN_Leaky, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# Res_block_body模块
class Res_block_body(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Res_block_body, self).__init__()
        self.out_channels = out_channels
        self.conv1 = DarknetConv2D_BN_Leaky(in_channels, out_channels, 3)
        self.conv2 = DarknetConv2D_BN_Leaky(out_channels // 2, out_channels // 2, 3)
        self.conv3 = DarknetConv2D_BN_Leaky(out_channels // 2, out_channels // 2, 3)
        self.conv4 = DarknetConv2D_BN_Leaky(out_channels, out_channels, 1)

        # 2×2的最大池化层
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):

        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route1
        route1 = x
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, self.out_channels // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route2
        route2 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route2], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat1 = x
        x = torch.cat([route1, x], dim=1)
        feat2 = self.maxpool(x)
        return feat1,feat2

# CSPdarknet53模块
class CSPDarkNet53(nn.Module):
    def __init__(self):
        super(CSPDarkNet53, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        self.conv1 = DarknetConv2D_BN_Leaky(3, 32, kernel_size=3, stride=2)
        self.conv2 = DarknetConv2D_BN_Leaky(32, 64, kernel_size=3, stride=2)

        self.res_block_body1 =  Res_block_body(64, 64)
        self.res_block_body2 =  Res_block_body(128, 128)
        self.res_block_body3 =  Res_block_body(256, 256)
        self.conv3 = DarknetConv2D_BN_Leaky(512, 512, kernel_size=3)
        self.num_features = 1

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _    = self.res_block_body1(x)
        x, _    = self.res_block_body2(x)
        x, Feat1    = self.res_block_body3(x)
        x = self.conv3(x)
        Feat2 = x
        return Feat1,Feat2

#总的backbone
def darknet53_tiny(pretrained):
    model = CSPDarkNet53()
    if pretrained:
        model.load_state_dict(torch.load("model_data/CSPdarknet53_tiny_backbone_weights.pth"))
    return model

