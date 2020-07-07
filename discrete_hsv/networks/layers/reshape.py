from torch import nn


class Reshape(nn.Module):
    def __init__(self, shape, **kwargs):
        self.shape = shape
        super(Reshape, self).__init__(**kwargs)

    def forward(self, input):
        return input.view(self.shape)
