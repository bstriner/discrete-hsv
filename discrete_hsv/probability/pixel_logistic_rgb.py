from torch import nn
import torch
from .gumbel import gumbel_softmax_sample
import numpy as np


class PixelLogisticRGBSampler(nn.Module):
    def __init__(self, k):
        super(PixelLogisticRGBSampler, self).__init__()
        self.k = k
        self.softplus = nn.Softplus()

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), 2, 3,
                             inputs.size(2), inputs.size(3))
        mu = inputs[:, 0]
        logprecision = inputs[:, 1]
        precision = torch.exp(logprecision)
        samples = torch.rand(size=mu.size(), device=mu.device)
        samples = torch.log(samples / (1-samples))
        samples = mu + (samples/precision)
        samples = (samples+1.0)*((self.k-1) / 2.0)
        samples = torch.round(samples).long()
        samples = torch.clamp(samples, 0, self.k-1)  # todo: check inclusive
        assert samples.size(1) == 3
        return samples


def logminusexp(b, a):
    #c= b.detach()
    #return torch.log(torch.exp(b-c)-torch.exp(a-c))+c
    #return torch.log(1.-torch.exp(a-b))+b
    #c = torch.max(b,a)
    # return torch.log(torch.exp(b-c)-torch.exp(a-c))+c
    return torch.log(torch.exp(b)-torch.exp(a)+1e-12)


class PixelLogisticRGBLoss(nn.Module):
    def __init__(self, k):
        super(PixelLogisticRGBLoss, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, input, target):
        #print(f"Loss input: {input.size()}, target: {target.size()}")
        assert input.size(1) == 6
        input = input.view(input.size(0), 2, 3,
                           input.size(2), input.size(3))
        target_float = (target.float()*2.0/(self.k-1)) - 1.0
        q = (0.5*2.0)/(self.k-1)
        mu = input[:, 0]
        #mu = mu/1000.+ target_float
        logprecision = input[:, 1]
        #print("Q:{}".format(q))
        #print("target {}->{}".format(torch.min(target_float).item(), torch.max(target_float).item()))
        precision = torch.exp(logprecision)
        #precision = self.softplus(logprecision)/1000.+500+1e-4
        #print("min: {}, max: {}".format(torch.min(precision).item(), torch.max(precision).item()))
        is_first = torch.eq(target, 0)
        is_last = torch.eq(target, self.k-1)
        ll_first = -self.softplus(-(target_float-mu+q)*precision)
        ll_last = -self.softplus((target_float-mu-q)*precision)
        ll_mid = logminusexp(
            -self.softplus(-(target_float-mu+q)*precision),
            -self.softplus(-(target_float-mu-q)*precision)
        )
        ll = torch.where(
            is_first,
            ll_first,
            torch.where(
                is_last,
                ll_last,
                ll_mid
            )
        )
        nll = - ll
        if not torch.all(torch.isfinite(nll)):
            #print("MAX: {}".format(torch.max(precision).detach().cpu().numpy()))
            # np.savez(
            #    'dump.npz', input=input.detach().cpu(), target=target.detach().cpu()
            # )
            raise ValueError()
        assert nll.size(1) == 3
        nll = torch.sum(nll, dim=1)
        return nll


class PixelLogisticRGB(nn.Module):
    def __init__(self, k):
        super(PixelLogisticRGB, self).__init__()
        self.k = k
        self.sampler = PixelLogisticRGBSampler(k=self.k)
        self.loss = PixelLogisticRGBLoss(k=self.k)
        self.dimensions = 2*3
        self.name = 'PixelLogisticRGB'
