import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, input_ch=3, output_ch=64, activf=nn.ReLU, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(output_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv_block = nn.Sequential(
            self.conv1,
            activf(inplace=True),
            self.conv2,
            activf(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    def __init__(self, input_ch=64, output_ch=32, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(input_ch, output_ch, 2, 2, bias=bias)
        self.conv_block = nn.Sequential(self.conv)

    def forward(self, x):
        return self.conv_block(x)


class UNetModule(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()

        # Encoder
        self.conv1 = ConvBlock(input_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, 2* base_ch)
        self.conv3 = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4 = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5 = ConvBlock(8 * base_ch, 16 * base_ch)

        # Decoder
        self.upconv1 = UpConv(16 * base_ch, 8 * base_ch)
        self.conv6 = ConvBlock(16 * base_ch, 8 * base_ch)
        self.upconv2 = UpConv(8 * base_ch, 4 * base_ch)
        self.conv7 = ConvBlock(8 * base_ch, 4 * base_ch)
        self.upconv3 = UpConv(4 * base_ch, 2 * base_ch)
        self.conv8 = ConvBlock(4 * base_ch, 2 * base_ch)
        self.upconv4 = UpConv(2 * base_ch, base_ch)
        self.conv9 = ConvBlock(2 * base_ch, base_ch)

        self.outconv = nn.Conv2d(base_ch, output_ch, 1, bias=True)

    def forward(self, x):

        x1 = self.conv1(x)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2(x)
        x = F.max_pool2d(x2, 2, 2)

        x3 = self.conv3(x)
        x = F.max_pool2d(x3, 2, 2)

        x4 = self.conv4(x)
        x = F.max_pool2d(x4, 2, 2)

        x = self.conv5(x)
        x = self.upconv1(x)
        x = torch.cat((x4, x), dim=1)

        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x3, x), dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x2, x), dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        x = torch.cat((x1, x), dim=1)

        x = self.conv9(x)
        x = self.outconv(x)

        return x


class UNet(UNetModule):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=None):
        super().__init__(input_ch, output_ch, base_ch)

    def forward(self, x):
        return [super().forward(x)]


class WNet(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=None):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, output_ch, base_ch)

    def forward(self, x):
        first_x = self.first_u(x)
        first_x_sig = torch.sigmoid(first_x)
        second_x = self.second_u(first_x_sig)
        return first_x, second_x


class RRUNet(nn.Module):
    """Mosinska et al. approach (without topology loss)"""

    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__()
        self.unet_module = UNetModule(input_ch+output_ch, output_ch, base_ch)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []
        x_size = x.size()
        zero_maps_size = (
            x_size[0],
            3,  # only refine AV3
            x_size[2],
            x_size[3],
        )
        zero_maps = torch.zeros(zero_maps_size).to(x.device)
        x_maps = torch.cat((x, zero_maps), dim=1)

        pred = self.unet_module(x_maps)
        predictions.append(pred)

        for _ in range(self.num_iterations):
            pred = torch.sigmoid(pred)
            x_maps = torch.cat((x, pred[:, :3, :, :]), dim=1)
            pred = self.unet_module(x_maps)
            predictions.append(pred[:, :3, :, :])

        return predictions


class SCUNet(nn.Module):
    """
    SCUNet (Successive Cascaded UNet) with 4 cascaded UNets.
    For UNet 2-4, input = original_image + previous_network_output (concatenated along channel dimension)
    The previous_network_output is expected to have 2 channels (A and V).

    Args:
        input_ch: Number of input channels (e.g., 3 for RGB)
        output_ch: Number of output channels (should be 2 for A and V)
        base_ch: Base number of channels for UNet modules
    """

    def __init__(self, input_ch=3, output_ch=2, base_ch=64):
        super().__init__()

        # First UNet: takes original image as input
        self.unet1 = UNetModule(input_ch, output_ch, base_ch)

        # Subsequent UNets: take original image + previous output (2 channels) as input
        # So input channels = input_ch + output_ch
        self.unet2 = UNetModule(input_ch + output_ch, output_ch, base_ch)
        self.unet3 = UNetModule(input_ch + output_ch, output_ch, base_ch)
        self.unet4 = UNetModule(input_ch + output_ch, output_ch, base_ch)

    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape (B, input_ch, H, W)

        Returns:
            List of predictions from all 4 UNets, each of shape (B, output_ch, H, W)
        """
        predictions = []

        # First UNet: only original image
        pred1 = self.unet1(x)
        predictions.append(pred1)

        # Apply sigmoid to get probability maps for A and V
        pred1_sigmoid = torch.sigmoid(pred1)

        # Second UNet: original image + previous output (2 channels)
        input2 = torch.cat([x, pred1_sigmoid], dim=1)  # Shape: (B, input_ch+2, H, W)
        pred2 = self.unet2(input2)
        predictions.append(pred2)

        # Third UNet: original image + previous output (2 channels)
        pred2_sigmoid = torch.sigmoid(pred2)
        input3 = torch.cat([x, pred2_sigmoid], dim=1)
        pred3 = self.unet3(input3)
        predictions.append(pred3)

        # Fourth UNet: original image + previous output (2 channels)
        pred3_sigmoid = torch.sigmoid(pred3)
        input4 = torch.cat([x, pred3_sigmoid], dim=1)
        pred4 = self.unet4(input4)
        predictions.append(pred4)

        return predictions


class RRWNetAll(nn.Module):
    """Network with all channels refined using a second recurrent
    UNet.
    """
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(3, 3, base_ch)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        pred_1 = torch.sigmoid(pred_1)

        pred_2 = self.second_u(pred_1[:, :3, :, :])
        predictions.append(pred_2)

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = self.second_u(pred_2)
            predictions.append(pred_2)

        return predictions


class RRWNet(RRWNetAll):
    """RRWNetAll but refining only A/V maps.
    Proposed in the paper.
    """

    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__(input_ch, output_ch, base_ch, num_iterations)
        self.second_u = UNetModule(output_ch, 2, base_ch)

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        bv_logits = pred_1[:, 2:3, :, :]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 2:3, :, :]

        pred_2 = self.second_u(pred_1)
        predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((pred_2, bv), dim=1)
            pred_2 = self.second_u(pred_2)
            predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        return predictions
