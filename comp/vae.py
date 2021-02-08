"""From Lossy Image Compression with Compressive Autoencoders"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # TODO(Todd): determine padding amount.
        # From Figure 2: the second convolution layer
        # nn.Conv2d(64, 128, 5, stride=2) would have an output
        # torch.Size([1, 128, 62, 62]).
        # This means that with padding=1, this residual layer have
        # an output of shape torch.Size([1, 128, 62, 62]).
        # Without padding, this residual layer will break.
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=1
        )
        self.leaky = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.conv2(x)
        x += residual
        return x


class Encoder(nn.Module):
    """Compression encoder

    Two initial layers of convolution, downsampling with stride 2,
    followed by three residual blocks, and finally, a convolution layer.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.res1 = ResBlock(128, 128, kernel_size=3)
        self.res2 = ResBlock(128, 128, kernel_size=3)
        self.res3 = ResBlock(128, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 96, kernel_size=5, stride=2)
        self.leaky = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.conv3(x)
        return x.round().int()


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        # TODO(Todd)
        raise NotImplementedError()
