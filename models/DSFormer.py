# !/usr/bin/env python
# -*-coding:utf-8 -*-


import numpy as np
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(p=0.1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(p=0.1),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True)
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class SingleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, 1, padding=1 if kernel_size == 3 else 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class DSSA(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads, hidden_channel=None, reduce_ratio=2, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        if not hidden_channel:
            hidden_channel = out_channel * 2

        self.head_dim = hidden_channel // num_heads
        self.num_heads = num_heads
        self.scale = (hidden_channel // num_heads) ** -0.5

        self.create_q = SingleConv(in_channel, hidden_channel, kernel_size=1)
        self.create_kv = SingleConv(in_channel, 2 * hidden_channel, kernel_size=1)
        # self.create_v = SingleConv(in_channel, hidden_channel, kernel_size=1)

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(2 * hidden_channel, 2 * hidden_channel, kernel_size=reduce_ratio, stride=reduce_ratio),
            nn.BatchNorm2d(2 * hidden_channel),
            nn.ReLU()
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_channel, hidden_channel)
        self.proj_drop = nn.Dropout(proj_drop)

        self.hidden_channel = hidden_channel

        self.final_conv = SingleConv(hidden_channel, out_channel, kernel_size=1)

    def forward(self, x):
        b, _, h, w = x.shape

        n = h * w

        q = self.create_q(x)
        kv = self.create_kv(x)
        # v = self.create_v(x)

        q = q.flatten(-2).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = self.reduce_conv(kv).flatten(-2).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attention = (q * self.scale) @ k.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = (attention @ v).transpose(1, 2).reshape(b, n, self.hidden_channel)
        x = self.proj(x)
        x = self.proj_drop(x).reshape(b, self.hidden_channel, h, w)

        return self.final_conv(x)

class TraditionalAttention(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads, hidden_channel=None, attn_drop=0., proj_drop=0.):
        super().__init__()


        if not hidden_channel:
            hidden_channel = out_channel
        self.hidden_channel = hidden_channel
        self.head_dim = hidden_channel // num_heads
        self.num_heads = num_heads

        self.scale = self.head_dim ** -0.5

        self.create_q = nn.Conv2d(in_channel, hidden_channel, kernel_size=1)
        self.create_k = nn.Conv2d(in_channel, hidden_channel, kernel_size=1)
        self.create_v = nn.Conv2d(in_channel, hidden_channel, kernel_size=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_channel, hidden_channel)
        self.proj_drop = nn.Dropout(proj_drop)

        self.final_conv = SingleConv(hidden_channel, out_channel, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, _, h, w = x.shape
        n = h * w

        q = self.create_q(x)
        k = self.create_k(x)
        v = self.create_v(x)

        q = q.flatten(-2).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.flatten(-2).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.flatten(-2).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention = torch.matmul(q, k.transpose(-1, -2))
        attention = attention / self.scale
        attention = self.softmax(attention)

        attention = self.attn_drop(attention)
        x = torch.matmul(attention, v).transpose(1, 2).reshape(b, n, self.hidden_channel).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x).reshape(b, self.hidden_channel, h, w)

        del attention
        return self.final_conv(x)


class DFFN(nn.Module):
    def __init__(self, c1):
        super(DFFN, self).__init__()

        c2 = 4 * c1

        self.conv1 = SingleConv(c1, c2, kernel_size=1)
        self.conv2 = SingleConv(c2, c1, kernel_size=1)

        self.depth_wise_conv = nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c1)
        self.norm = nn.BatchNorm2d(c2)

        self.activate = nn.GELU()

        self.fc = nn.Sequential(
            nn.Linear(c2, c2),
            nn.LayerNorm(c2)
        )

    def forward(self, x):
        b, _, h, w = x.shape

        x = self.conv1(x)
        residual = x
        x2 = x.flatten(-2).transpose(1, 2)

        x1 = self.depth_wise_conv(x)
        x1 = self.norm(x1 + residual)

        x2 = self.fc(x2)
        x2 = x2.view(b, h, w, -1).contiguous().permute(0, 3, 1, 2)

        x = self.activate(x1 + x2)

        return self.conv2(x)

class LeFF(nn.Module):
    def __init__(self, c1):
        super().__init__()
        c2 = 4 * c1
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1)
        self.depth_wise_conv = nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2)
        self.activate = nn.GELU()
        self.conv2 = nn.Conv2d(c2, c1, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.depth_wise_conv(x)
        x = self.activate(x)
        x = self.conv2(x)
        x = x + residual

        return x

class Enhanced_MixFFN(nn.Module):
    def __init__(self, c1):
        super().__init__()
        c2 = 4 * c1
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1)
        self.depth_wise_conv = nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2)
        self.activate = nn.GELU()
        self.conv2 = nn.Conv2d(c2, c1, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(c2)
        self.norm2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.depth_wise_conv(x)
        x = self.activate(self.norm1(x + self.conv1(residual)))
        x = self.conv2(x)
        x = self.norm2(x + residual)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, reduce_ratio, attention_mechanism, feedforward):
        super().__init__()
        if attention_mechanism == 'DSSA':
            self.attention = DSSA(
                in_channel=in_channel, out_channel=out_channel, num_heads=4, reduce_ratio=reduce_ratio
            )
        else:
            self.attention = TraditionalAttention(
                in_channel=in_channel, out_channel=out_channel, num_heads=4
            )
        self.residual_conv = SingleConv(in_channel, out_channel)

        if feedforward == 'DFFN':
            self.mlp = DFFN(out_channel)
        elif feedforward == 'LeFF':
            self.mlp = LeFF(out_channel)
        else:
            self.mlp = Enhanced_MixFFN(out_channel)

        self.norm1 = nn.BatchNorm2d(in_channel)
        self.norm2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        residual = self.residual_conv(x)
        tx = residual + self.attention(self.norm1(x))
        mx = tx + self.mlp(self.norm2(tx))
        return mx


class DoubleAttention(nn.Module):
    def __init__(self, in_channel, out_channel, reduce_ratio, attention_mechanism, feedforward):
        super().__init__()
        self.layers = nn.Sequential(
            AttentionBlock(in_channel, out_channel, reduce_ratio, attention_mechanism, feedforward),
            AttentionBlock(out_channel, out_channel, reduce_ratio, attention_mechanism, feedforward)
            )


    def forward(self, x):
        return self.layers(x)


class UpSample(nn.Module):
    def __init__(self, in_channel, mode='bilinear'):
        super().__init__()

        if mode:
            self.layers = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        else:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.LeakyReLU(0.1, False)
            )

    def forward(self, x):
        return self.layers(x)


class BridgeBlock(nn.Module):
    def __init__(self, bridge_setting, dropout=0.):
        super().__init__()
        filters = [16 * 2 ** i for i in range(4)]

        self.bridge_setting = bridge_setting

        self.fuse0 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, False)
        )

        self.fuse1 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.1, False)
        )

        self.fuse2 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[2], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(0.1, False)
        )

        self.fuse3 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[3], kernel_size=8, stride=8, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.LeakyReLU(0.1, False)
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        if len(bridge_setting) == 1:
            tmp_filter = filters[int(bridge_setting) - 1]
        else:
            tmp_filter = sum(filters[int(i) - 1] for i in bridge_setting)

        self.conv1 = nn.Conv2d(tmp_filter, 4 * tmp_filter, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4 * tmp_filter, tmp_filter, kernel_size=1, bias=False)

        self.defuse0 = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(0.1, False)
        )

        self.defuse1 = nn.Sequential(
            nn.Conv2d(filters[1], filters[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.LeakyReLU(0.1, False)
        )

        self.defuse2 = nn.Sequential(
            nn.Conv2d(filters[2], filters[2], kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.LeakyReLU(0.1, False)
        )

        self.defuse3 = nn.Sequential(
            nn.Conv2d(filters[3], filters[3], kernel_size=8, stride=8, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.LeakyReLU(0.1, False)
        )

        self.filters = filters

    def forward(self, x0, x1, x2, x3):
        x0 = self.fuse0(x0)
        x1 = self.fuse1(x1)
        x2 = self.fuse2(x2)
        x3 = self.fuse3(x3)

        tmp_x = [x0, x1, x2, x3]
        if len(self.bridge_setting) == 1:
            x = tmp_x[int(self.bridge_setting) - 1]
        else:
            x = torch.cat([tmp_x[int(i) - 1] for i in self.bridge_setting], dim=1)

        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)

        if len(self.bridge_setting) == 1:
            tmp_x[int(self.bridge_setting) - 1] = x
        else:
            m_x = torch.split(x, [self.filters[int(i) - 1] for i in self.bridge_setting], dim=1)
            for idx, i in enumerate(self.bridge_setting):
                tmp_x[int(i) - 1] = m_x[idx]

        x0, x1, x2, x3 = tmp_x

        x0 = self.defuse0(x0)
        x1 = self.defuse1(x1)
        x2 = self.defuse2(x2)
        x3 = self.defuse3(x3)

        return x0, x1, x2, x3


class DSFormer(nn.Module):
    def __init__(self, in_channel, out_channel, init_ratio, bridge_setting, attention_mechanism, feedforward):
        super().__init__()
        filters = [16 * 2 ** i for i in range(5)]
        reduce_ratio = [init_ratio * 2 ** i for i in range(5)]
        reduce_ratio.reverse()

        self.attention_mechanism = attention_mechanism
        self.feedforward = feedforward

        self.encoder1 = self.make_block(in_channel, filters[0], reduce_ratio[0])
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            self.make_block(filters[0], filters[1], reduce_ratio[1])
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            self.make_block(filters[1], filters[2], reduce_ratio[2])
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2),
            self.make_block(filters[2], filters[3], reduce_ratio[3])
        )

        self.middle = nn.Sequential(
            nn.MaxPool2d(2),
            self.make_block(filters[3], filters[4], reduce_ratio[4])
        )
        if bridge_setting != '0':
            self.bridge = BridgeBlock(bridge_setting)

        self.decoder = UpSample(filters[4])
        self.decoder1 = nn.Sequential(
            DoubleConv(filters[4] + filters[3], filters[3]),
            UpSample(filters[3])
        )
        self.decoder2 = nn.Sequential(
            DoubleConv(filters[3] + filters[2], filters[2]),
            UpSample(filters[2])

        )
        self.decoder3 = nn.Sequential(
            DoubleConv(filters[2] + filters[1], filters[1]),
            UpSample(filters[1])

        )
        self.decoder4 = DoubleConv(filters[1] + filters[0], filters[0])
        self.out_conv = nn.Conv2d(filters[0], out_channel, kernel_size=1)

        self.bridge_setting = bridge_setting

    def forward(self, x):
        x1 = self.encoder1(x)  # B C H W
        x2 = self.encoder2(x1)  # B 2C H/2 W/2
        x3 = self.encoder3(x2)  # B 4C H/4 W/4
        x4 = self.encoder4(x3)  # B 8C H/8 W/8

        x = self.middle(x4)

        if self.bridge_setting != '0':
            x1, x2, x3, x4 = self.bridge(x1, x2, x3, x4)

        x = self.decoder1(torch.cat((self.decoder(x), x4), dim=1))
        x = self.decoder2(torch.cat((x, x3), dim=1))
        x = self.decoder3(torch.cat((x, x2), dim=1))
        x = self.decoder4(torch.cat((x, x1), dim=1))

        x = self.out_conv(x)

        return x

    def make_block(self, in_channel, out_channel, init_ratio=None):
        if self.attention_mechanism:
            return DoubleAttention(in_channel, out_channel, init_ratio, self.attention_mechanism, self.feedforward)
        else:
            return DoubleConv(in_channel, out_channel)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=="__main__":
    a = torch.rand(2, 3, 256, 256)
    net = DSFormer(
        3, 3, init_ratio=2, bridge_setting='123', attention_mechanism=None, feedforward='DFFN'
    )
    
    print(net)
    print(np.round(count_parameters(net) / 1e6, 2))
    t = net(a)
    print(t.shape)
