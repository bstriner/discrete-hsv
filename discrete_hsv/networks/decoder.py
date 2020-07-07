import torch
from torch import nn
from .layers.reshape import Reshape


class Decoder(nn.Module):

    def __init__(self, z_dim, distribution):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=8*8*128),
            nn.LeakyReLU(negative_slope=0.2),
            Reshape(shape=(-1, 128, 8, 8)),
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=distribution.dimensions,
                      kernel_size=1, stride=1, padding=0)
        )
        #self.weights_init()

    def forward(self, inputs):
        output = self.net(inputs)
        #print(f"decoder input: {inputs.size()}, output: {output.size()}")
        assert output.size(2) == 32
        assert output.size(3) == 32
        return output
