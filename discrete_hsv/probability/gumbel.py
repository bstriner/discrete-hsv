import torch


def gumbel_sample(size, device):
    #gumbels = torch.rand(size=size, device=device)
    #gumbels=- torch.log(1-torch.log(rand))
    gumbels = - \
        torch.empty(size=size, device=device,
                    memory_format=torch.legacy_contiguous_format).exponential_().log()
    return gumbels


def gumbel_softmax_logits(inputs):
    logits = inputs + gumbel_sample(
        size=inputs.size(),
        device=inputs.device)
    return logits


def gumbel_softmax_sample(inputs, axis=-1):
    return torch.argmax(gumbel_softmax_logits(inputs), axis=axis)
