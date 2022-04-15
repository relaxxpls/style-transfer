import torch.nn as nn


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
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # ? size of feature maps
        self.num_features = 64

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, self.num_features, 7, padding=3),
            nn.InstanceNorm2d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_features, self.num_features * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(self.num_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.num_features * 2, self.num_features * 4, 3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(self.num_features * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(self.num_features * 4),
            ResidualBlock(self.num_features * 4),
            ResidualBlock(self.num_features * 4),
            ResidualBlock(self.num_features * 4),
            ResidualBlock(self.num_features * 4),
            ResidualBlock(self.num_features * 4),
            ResidualBlock(self.num_features * 4),
            ResidualBlock(self.num_features * 4),
            ResidualBlock(self.num_features * 4),
            nn.ConvTranspose2d(
                self.num_features * 4,
                self.num_features * 2,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(self.num_features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                self.num_features * 2,
                self.num_features,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)
