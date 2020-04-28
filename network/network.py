from torch import nn
import numpy as np


class ExtractNet(nn.Module):

    def __init__(self, features):
        super(ExtractNet, self).__init__()
        self.features = features
        self.need_layer = ['3', '8', '15', '22']

    def forward(self, x):
        result = []
        for layer, module in self.features._modules.items():
            x = module(x)
            if layer in self.need_layer:
                result.append(x)
        return np.array(result)


def baseConvLayer(in_channel, out_channel, kernel_size=3, stride=1, unSample=None, bn=True, relu=True):
    padding = kernel_size // 2
    layers = []
    if unSample is not None:
        # 放缩
        layers.append(nn.UpsamplingBilinear2d(scale_factor=unSample))
    # 镜像填充
    layers.append(nn.ReflectionPad2d(padding))
    # 基础卷积
    layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride))
    if bn:
        # bn
        layers.append(nn.BatchNorm2d(out_channel))
    if relu:
        # relu
        layers.append(nn.RReLU())
    return layers


class ResidualNet(nn.Module):

    def __init__(self, channels):
        super(ResidualNet, self).__init__()
        self.conv = nn.Sequential(
            *baseConvLayer(channels, channels),
            *baseConvLayer(channels, channels, relu=False)
        )

    def forward(self, x):
        return self.conv(x) + x


class StyleTransformNet(nn.Module):

    def __init__(self, base=32):
        super(StyleTransformNet, self).__init__()
        self.downSample = nn.Sequential(
            *baseConvLayer(3, base, kernel_size=9),
            *baseConvLayer(base, 2 * base, stride=2),
            *baseConvLayer(2 * base, 4 * base, stride=2)
        )
        self.residual = nn.Sequential(
            *[ResidualNet(base * 4) for _ in range(4)]
        )
        self.unSample = nn.Sequential(
            *baseConvLayer(4 * base, 2 * base, unSample=2),
            *baseConvLayer(2 * base, base, unSample=2),
            *baseConvLayer(base, 3, kernel_size=9, bn=False, relu=False)
        )

    def forward(self, originImage):
        downSample = self.downSample(originImage)
        residualFeature = self.residual(downSample)
        upSampleFeature = self.unSample(residualFeature)
        return upSampleFeature
