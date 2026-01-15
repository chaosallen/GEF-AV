"""
Channel and Spatial Attention Network (CSNet)
Safe standard implementation (no inplace / autograd conflict)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Basic Ops ----------
def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def upsample(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def init_weights(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# ---------- Encoder ----------
class ResEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv(x)
        return self.relu(out + res)


# ---------- Decoder ----------
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)


# ---------- Spatial Attention ----------
class SpatialAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.query = nn.Conv2d(in_ch, in_ch // 8, kernel_size=(1, 3), padding=(0, 1))
        self.key   = nn.Conv2d(in_ch, in_ch // 8, kernel_size=(3, 1), padding=(1, 0))
        self.value = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = self.softmax(torch.matmul(q, k))
        v = self.value(x).view(B, C, -1)
        out = torch.matmul(v, attn.permute(0, 2, 1))
        return out.view(B, C, H, W)


# ---------- Channel Attention ----------
class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = x.view(B, C, -1)
        k = x.view(B, C, -1).permute(0, 2, 1)
        attn = self.softmax(torch.matmul(q, k))
        v = x.view(B, C, -1)
        out = torch.matmul(attn, v)
        return out.view(B, C, H, W)


# ---------- Affinity Attention ----------
class AffinityAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.spatial = SpatialAttention(in_ch)
        self.channel = ChannelAttention()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        sa = self.spatial(x)
        ca = self.channel(x)
        # single residual, safe
        return x + self.gamma * (sa + ca)


# ---------- CSNet ----------
class CSNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.enc0 = ResEncoder(in_channels, 32)
        self.enc1 = ResEncoder(32, 64)
        self.enc2 = ResEncoder(64, 128)
        self.enc3 = ResEncoder(128, 256)
        self.enc4 = ResEncoder(256, 512)

        self.pool = downsample()
        self.attn = AffinityAttention(512)

        self.up4 = upsample(512, 256)
        self.dec4 = Decoder(512, 256)

        self.up3 = upsample(256, 128)
        self.dec3 = Decoder(256, 128)

        self.up2 = upsample(128, 64)
        self.dec2 = Decoder(128, 64)

        self.up1 = upsample(64, 32)
        self.dec1 = Decoder(64, 32)

        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

        init_weights(self)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.attn(e4)

        d4 = self.dec4(torch.cat([self.up4(b), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))

        return self.final(d1)
