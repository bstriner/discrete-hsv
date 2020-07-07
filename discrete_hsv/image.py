import torch
from .vae import VAE
from ignite.engine import Engine, Events
from torch.utils import data
from torchvision.datasets.cifar import CIFAR10
import torch
from torch import optim
import numpy as np
import argparse
import os
from ignite.metrics.running_average import RunningAverage
from .ignite_util import get_value, print_logs, print_times, create_plots, handle_exception
from torch import nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from .probability.kl import calc_kl
from torchvision import transforms
from torchvision.utils import save_image

class ImageTransform(object):
    def __call__(self, pic):
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))

        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        return img


class ImageScaleInputRGB(nn.Module):
    def forward(self, inputs):
        return (inputs.float().div(255)*2)-1

class ImageScaleInputHSV(nn.Module):
    def forward(self, inputs):
        raise NotImplementedError()


class ImageScaleOutput(nn.Module):
    def forward(self, inputs):
        return inputs.float().div(255)


def cifar10(
        dataroot,
        workers,
        batch_size):
    dataset = CIFAR10(root=dataroot, download=True, transform=ImageTransform())
    # transform=transforms.ToTensor())
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=workers, drop_last=True)
    return loader
