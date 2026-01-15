# 2D-Unet Model taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
import torch
import torch.nn as nn

# MultiAttention UNet
#Channel Attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

#Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_input = x
        max_out, _ = torch.max(x_input, dim=1, keepdim=True)
        avg_out = torch.mean(x_input, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x))
        return x_input * attn

#Self Attention
class SelfAttention2D(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention2D, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out

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
        self.se = SEBlock(out_size)
        #self.dropout = nn.Dropout2d(p=0.1)


    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        x=self.se(x)
        #x = self.dropout(x)
        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.spatial_attention = SpatialAttention()
    def forward(self, inputs0, *input):
        # print(self.n_concat)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        outputs0 = self.conv(outputs0)
        outputs0 = self.spatial_attention(outputs0)
        return outputs0

class MAUNet(nn.Module):

    def __init__(self, in_channels, n_classes, channels=64,channels1st=2, is_deconv=True, is_batchnorm=True):
        super(MAUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.channels = channels
        self.n_classes=n_classes
        self.channels1st=channels1st

        # downsampling
        self.conv1 = unetConv2(self.in_channels, self.channels, self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(self.channels, self.channels*2, self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(self.channels*2, self.channels*4, self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(self.channels*4, self.channels*8, self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center1 = unetConv2(self.channels*8, self.channels*16, self.is_batchnorm)
        self.attn1 = SelfAttention2D(self.channels * 16)

        # upsampling
        self.up_concat4 = unetUp(self.channels*16, self.channels*8, self.is_deconv)
        self.up_concat3 = unetUp(self.channels*8,self.channels*4, self.is_deconv)
        self.up_concat2 = unetUp(self.channels*4, self.channels*2, self.is_deconv)
        self.up_concat1 = unetUp(self.channels*2, self.channels, self.is_deconv)

        self.outconv1 = nn.Conv2d(self.channels,self.channels1st, 3, padding=1)

        # downsampling
        self.conv5 = unetConv2(self.channels1st+self.in_channels, self.channels, self.is_batchnorm)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = unetConv2(self.channels, self.channels * 2, self.is_batchnorm)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2)

        self.conv7 = unetConv2(self.channels * 2, self.channels * 4, self.is_batchnorm)
        self.maxpool7 = nn.MaxPool2d(kernel_size=2)

        self.conv8 = unetConv2(self.channels * 4, self.channels * 8, self.is_batchnorm)
        self.maxpool8 = nn.MaxPool2d(kernel_size=2)

        self.center2 = unetConv2(self.channels * 8, self.channels * 16, self.is_batchnorm)
        self.attn2 = SelfAttention2D(self.channels * 16)

        # upsampling
        self.up_concat8 = unetUp(self.channels * 16, self.channels * 8, self.is_deconv)
        self.up_concat7 = unetUp(self.channels * 8, self.channels * 4, self.is_deconv)
        self.up_concat6 = unetUp(self.channels * 4, self.channels * 2, self.is_deconv)
        self.up_concat5 = unetUp(self.channels * 2, self.channels, self.is_deconv)

        self.outconv2 = nn.Conv2d(self.channels, self.n_classes, 3, padding=1)


    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center1 = self.center1(maxpool4)
        center1 = self.attn1(center1)


        up4 = self.up_concat4(center1, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)

        output1 = self.outconv1(up1)
        output1a = torch.cat([output1, inputs], 1)

        conv5 = self.conv5(output1a)
        maxpool5 = self.maxpool5(conv5)

        conv6 = self.conv6(maxpool5)
        maxpool6 = self.maxpool6(conv6)

        conv7 = self.conv7(maxpool6)
        maxpool7 = self.maxpool7(conv7)

        conv8 = self.conv8(maxpool7)
        maxpool8 = self.maxpool8(conv8)

        center2 = self.center2(maxpool8)
        center2 = self.attn2(center2)

        up8 = self.up_concat8(center2, conv8)
        up7 = self.up_concat7(up8, conv7)
        up6 = self.up_concat6(up7, conv6)
        up5 = self.up_concat5(up6, conv5)

        output2=self.outconv2(up5)


        return output1,output2
