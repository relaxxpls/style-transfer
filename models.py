import torch.nn as nn
import torch.nn.functional as F


class PrintLayer(nn.Module):
    def __init__(self, title=None):
        super().__init__()
        self.title = title

    def forward(self, x):
        print(self.title, x.shape)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        x = x + self.conv_block(x)

        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            # Initial convolution block
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            # Downsampling
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            # Residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            # Upsampling
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            # Output layer
            nn.Conv2d(64, out_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.model(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, x):
        # ? x.shape: (batch_size, 1, 64, 64)
        x = self.model(x)
        # ? x.shape: (batch_size, 1, 6, 6)
        x = F.avg_pool2d(x, x.shape[2:])
        # ? x.shape: (batch_size, 1, 1, 1)
        x = x.flatten(0)

        return x
