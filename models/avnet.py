import torch
import torch.nn as nn
#AV
class Denseblock(nn.Module):
    """(convolution=> ReLU) * 6"""

    def __init__(self, channels,n):
        super().__init__()
        self.n=n
        for i in range(1, n + 1):
            convblock = Convblock(channels)
            setattr(self, 'convblock%d' % i, convblock)

    def forward(self, x):
        for i in range(1, self.n + 1):
            conv = getattr(self, 'convblock%d' % i)
            x = conv(x)
        return x
class Convblock(nn.Module):
    """(convolution=> ReLU) * 6"""

    def __init__(self, channels):
        super().__init__()
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1=self.conv1_3(x)
        x2=torch.cat([x, x1], 1)
        output=self.conv1_1(x2)
        return output
class Decoderblock(nn.Module):
    """(convolution=> ReLU) * 6"""

    def __init__(self, channels):
        super().__init__()
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x=torch.cat([x1, x2], 1)
        output=self.conv3_3(x)
        return output
class AVNet(nn.Module):

    def __init__(self, in_channels, n_classes, channels=64, is_deconv=True, is_batchnorm=False):
        super(AVNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.channels = channels
        self.n_classes=n_classes

        # downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.dense1 = Denseblock(channels,n=6)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.dense2 = Denseblock(channels, n=12)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.dense3 = Denseblock(channels, n=24)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2)

        self.dense4 = Denseblock(channels, n=16)

        self.upsample1= nn.UpsamplingBilinear2d(scale_factor=2)
        self.decode1= Decoderblock(channels)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decode2 = Decoderblock(channels)

        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decode3 = Decoderblock(channels)

        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decode4 = Decoderblock(channels)

        self.outconv = nn.Conv2d(self.channels, self.n_classes, 1)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        dense1 = self.dense1(maxpool1)
        avgpool1 = self.avgpool2(dense1)

        dense2 = self.dense1(avgpool1)
        avgpool2 = self.avgpool2(dense2)

        dense3 = self.dense1(avgpool2)
        avgpool3 = self.avgpool3(dense3)

        dense4 = self.dense1(avgpool3)

        unsample1= self.upsample1(dense4)
        decode1 = self.decode1(unsample1,dense3)

        unsample2 = self.upsample2(decode1)
        decode2 = self.decode2(unsample2, dense2)

        unsample3 = self.upsample3(decode2)
        decode3 = self.decode3(unsample3, dense1)

        unsample4 = self.upsample4(decode3)
        decode4 = self.decode4(unsample4, conv1)

        output = self.outconv(decode4)

        return output