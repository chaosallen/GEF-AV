import torch
import torch.nn as nn
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=3, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size


    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)



    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
class UNetpp(nn.Module):

    def __init__(self, in_channels, n_classes, channels=64, is_deconv=True, is_batchnorm=True,is_ds=True):
        super(UNetpp, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.channels = channels
        self.n_classes=n_classes
        self.is_ds = is_ds

        # downsampling
        self.conv00 = unetConv2(self.in_channels, self.channels, self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(self.channels,self.channels, self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(self.channels,self.channels, self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(self.channels, self.channels, self.is_batchnorm)


        # upsampling
        self.up_concat01 = unetUp(self.channels*2, self.channels, self.is_deconv)
        self.up_concat11 = unetUp(self.channels*2, self.channels, self.is_deconv)
        self.up_concat21 = unetUp(self.channels*2, self.channels, self.is_deconv)
        self.up_concat31 = unetUp(self.channels*2, self.channels, self.is_deconv)

        self.up_concat02 = unetUp(self.channels*3, self.channels, self.is_deconv)
        self.up_concat12 = unetUp(self.channels*3, self.channels, self.is_deconv)
        self.up_concat22 = unetUp(self.channels*3, self.channels, self.is_deconv)

        self.up_concat03 = unetUp(self.channels*4, self.channels,self.is_deconv)
        self.up_concat13 = unetUp(self.channels*4, self.channels, self.is_deconv)

        self.up_concat04 = unetUp(self.channels*5, self.channels,self.is_deconv)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(self.channels, n_classes, 1)
        self.final_2 = nn.Conv2d(self.channels, n_classes, 1)
        self.final_3 = nn.Conv2d(self.channels, n_classes, 1)
        self.final_4 = nn.Conv2d(self.channels, n_classes, 1)



    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4


        return final