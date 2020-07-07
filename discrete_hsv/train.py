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
from .image import ImageTransform, ImageScaleInputRGB, ImageScaleOutput, cifar10
CKPT_PREFIX = "networks"
FAKE_IMG_FNAME = "fake-{}.png"
REAL_IMG_FNAME = "real-{}.png"


def train_run(
    device,
    output_dir,
    dataroot,
    z_dim,
    distribution,
    epochs,
    loader,
    learning_rate,
    beta_1,
    sample_size,
    alpha,
    print_freq,
    **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    vae = VAE(
        z_dim=z_dim,
        distribution=distribution
    ).to(device=device)
    scaleinput = ImageScaleInputRGB()
    scaleoutput = ImageScaleOutput()

    optimizer = optim.Adam(
        vae.parameters(), lr=learning_rate, betas=(beta_1, 0.999))

    def draw_noise(n):
        return torch.randn(n, z_dim, device=device)

    def sample(mu, logsigmasq):
        return mu + (torch.exp(logsigmasq/2.) * torch.randn_like(logsigmasq))

    fixed_noise = draw_noise(sample_size)

    def step(engine, batch):
        real, _ = batch
        real = real.to(device)
        real_scaled = scaleinput(real)
        n = real.size(0)
        vae.zero_grad()
        vae.train()
        mu, logsigmasq = vae.encoder(real_scaled)
        z = sample(mu=mu, logsigmasq=logsigmasq)
        x_pred = vae.decoder(z)
        reconstruction = torch.sum(
            vae.distribution.loss(input=x_pred, target=real))/n
        kl = torch.mean(calc_kl(mu=mu, logsigmasq=logsigmasq, axis=-1))
        loss = reconstruction+kl
        loss.backward()
        optimizer.step()
        return {
            "kl": kl.item(),
            "reconstruction": reconstruction.item(),
            "loss": loss.item()
        }

    # ignite objects
    trainer = Engine(step)

    # attach running average metrics
    monitoring_metrics = ["kl", "reconstruction",
                          "loss"]
    for metric in monitoring_metrics:
        RunningAverage(alpha=alpha, output_transform=get_value(
            metric)).attach(trainer, metric)

    # Save generated images
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_fake_example(engine):
        vae.eval()
        fake = vae.decoder(fixed_noise)
        fake = vae.distribution.sampler(fake)
        fake = scaleoutput(fake)
        path = os.path.join(
            output_dir, FAKE_IMG_FNAME.format(engine.state.epoch)
        )
        save_image(fake.detach(), path)

    # Save real images
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_real_example(engine):
        img, _ = engine.state.batch
        img = scaleoutput(img)
        path = os.path.join(
            output_dir, REAL_IMG_FNAME.format(engine.state.epoch))
        save_image(img, path)

    # Progress bar
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(
        every=print_freq), print_logs(output_dir, max_epoch=epochs, loader=loader, pbar=pbar))

    # Saver
    checkpoint_handler = ModelCheckpoint(
        output_dir, CKPT_PREFIX, n_saved=10, require_empty=False)
    to_save = {
        # , 'args': Args(args)
        "vae": vae, "trainer": trainer, "optimizer": optimizer
    }
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save=to_save
    )

    # Timer
    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    # Print the timing
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              print_times(pbar=pbar, timer=timer))
    # Create plots
    trainer.add_event_handler(Events.EPOCH_COMPLETED, create_plots)
    # Handle ctrl-C or other exceptions
    trainer.add_event_handler(Events.EXCEPTION_RAISED,  handle_exception(
        handler=checkpoint_handler, to_save=to_save
    ))

    trainer.run(loader, max_epochs=epochs)


def train_args(output='output/vae', distribution='PixelSoftmax', epochs=100, z_dim=100):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", help="path to dataset", default='data')
    parser.add_argument(
        "--distribution", help="path to dataset", default=distribution)

    parser.add_argument("--workers", type=int, default=2,
                        help="number of data loading workers")

    parser.add_argument("--batch-size", type=int,
                        default=32, help="input batch size")
    parser.add_argument("--print-freq", type=int,
                        default=100, help="input batch size")

    parser.add_argument("--z-dim", type=int, default=z_dim,
                        help="size of the latent z vector")

    parser.add_argument("--epochs", type=int, default=epochs,
                        help="number of epochs to train for")

    parser.add_argument("--learning-rate", type=float, default=0.0003,
                        help="learning rate")

    parser.add_argument("--beta-1", type=float,
                        default=0.9, help="beta_1 for adam")

    parser.add_argument("--sample-size", type=int,
                        default=32, help="beta_1 for adam")

    parser.add_argument("--no-cuda", action="store_true", help="disables cuda")

    parser.add_argument("--output-dir", default=output,
                        help="directory to output images and model checkpoints")

    parser.add_argument("--alpha", type=float, default=0.98,
                        help="smoothing constant for exponential moving averages")
    args = parser.parse_args()
    return args


def train(**kwargs):
    args = train_args(**kwargs)
    device = "cpu" if (not torch.cuda.is_available()
                       or args.no_cuda) else "cuda:0"
    loader = cifar10(
        dataroot=args.dataroot,
        workers=args.workers,
        batch_size=args.batch_size
    )
    train_run(
        loader=loader,
        device=device,
        **vars(args)
    )
