from torch import nn
import torch
from .gumbel import gumbel_softmax_sample


class PixelSoftmaxSampler(nn.Module):
    def __init__(self, k):
        super(PixelSoftmaxSampler, self).__init__()
        self.k = k

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), self.k, 3,
                             inputs.size(2), inputs.size(3))
        inputs = gumbel_softmax_sample(inputs=inputs, axis=1)
        return inputs


class PixelSoftmaxLoss(nn.Module):
    def __init__(self, k):
        super(PixelSoftmaxLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.k = k

    def forward(self, input, target):
        input = input.view(input.size(0), self.k, 3,
                           input.size(2), input.size(3))
        loss = self.loss(input=input, target=target.long())
        loss = torch.sum(loss, dim=1)
        return loss


class PixelSoftmax(nn.Module):
    def __init__(self, k):
        super(PixelSoftmax, self).__init__()
        self.k = k
        self.sampler = PixelSoftmaxSampler(k=self.k)
        self.loss = PixelSoftmaxLoss(k=self.k)
        self.dimensions = self.k*3
        self.name = 'PixelSoftmax'
