import torch
from torch import nn
from .layers.reshape import Reshape


class Encoder(nn.Module):
    """ Discriminator network.

    Args:
        nf (int): Number of filters in the first conv layer.
    """

    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(shape=(-1, 128*8*8)),
            nn.Linear(in_features=128*8*8, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mu_projection = nn.Linear(in_features=512, out_features=z_dim)
        self.logsigmasq_projection = nn.Linear(in_features=512, out_features=z_dim)

        #self.weights_init()

    def forward(self, x):
        output = self.net(x)
        mu = self.mu_projection(output)
        logsigmasq = self.logsigmasq_projection(output)
        return mu, logsigmasq
