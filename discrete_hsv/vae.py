import torch
from torch import nn
from .networks.encoder import Encoder
from .networks.decoder import Decoder
from .probability.pixel_softmax import PixelSoftmax
from .probability.pixel_logistic_rgb import PixelLogisticRGB


class VAE(nn.Module):
    def __init__(self, z_dim, distribution):
        super(VAE, self).__init__()
        if distribution == 'PixelSoftmax':
            self.distribution = PixelSoftmax(k=256)
        elif distribution == 'PixelLogisticRGB':
            self.distribution = PixelLogisticRGB(k=256)
        else:
            raise ValueError()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, distribution=self.distribution)
