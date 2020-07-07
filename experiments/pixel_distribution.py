from discrete_hsv.probability.pixel_softmax import PixelSoftmax
from discrete_hsv.probability.pixel_logistic_rgb import PixelLogisticRGB
import torch
import tqdm
import numpy as np


class Raw(torch.nn.Module):
    def __init__(self, size):
        super(Raw, self).__init__()
        self.raw = torch.nn.Parameter(torch.FloatTensor().new(*size))
        torch.nn.init.normal_(self.raw)


def evaluate_distribution(distribution, n=64, iters=500000, device="cuda:0", h=32, w=32, lr=1e-2):
    #inputs = torch.IntTensor(n, 3).to(device=device).uniform_(0, distribution.k-1)
    inputs = torch.randint(low=0, high=distribution.k, size=(
        n, 3, h, w), device=device, dtype=torch.int32)
    raw = Raw(size=(n, distribution.dimensions, h, w)).to(device)
    opt = torch.optim.Adam(params=raw.parameters(), lr=lr)
    it = tqdm.tqdm(range(iters), desc='Training')
    for i in it:
        #np.savez(
        #    'dump-{}.npz'.format(i), input=raw.raw.detach().cpu(), target=inputs.detach().cpu()
        #)

        assert torch.all(torch.isfinite(raw.raw))
        raw.eval()
        raw.requires_grad = False
        samples = distribution.sampler(raw.raw)

        raw.train()
        raw.requires_grad = True
        raw.zero_grad()
        loss = distribution.loss(input=raw.raw, target=inputs)
        loss = torch.mean(loss)
        loss.backward()
        assert torch.all(torch.isfinite(raw.raw.grad))
        opt.step()

        acc = torch.eq(samples, inputs)
        pixelacc = torch.mean(acc.float()).item()
        totacc = torch.mean(torch.all(acc, dim=1).float()).item()
        it.desc = "{}: Loss: {}, Pred acc: {}, Accuracy: {}, {}".format(
            distribution.name, loss.detach().item(), torch.exp(-loss).item(), pixelacc, totacc)
    tqdm.tqdm.write("{}: Loss: {}, Accuracy: {}".format(
        distribution.name, loss.detach().item(), acc))


def main():
    no_cuda = False
    device = "cpu" if (not torch.cuda.is_available() or no_cuda) else "cuda:0"
    for distribution in [
        #PixelSoftmax(k=256),
        PixelLogisticRGB(k=256)
    ]:
        evaluate_distribution(distribution=distribution, device=device)


if __name__ == '__main__':
    main()
