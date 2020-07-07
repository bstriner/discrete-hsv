
import torch


def calc_kl(mu, logsigmasq, axis=-1):
    return 0.5 * torch.sum(torch.exp(logsigmasq) +
                           torch.square(mu) - 1. - logsigmasq, axis=axis)
