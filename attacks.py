import torch
import torch.nn.functional as F

import config


def fgsm(model, images, labels, eps=config.EPSILON):
    images = images.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(images), labels)
    (grad,) = torch.autograd.grad(loss, images)
    adv = images + eps * grad.sign()
    return adv.clamp(0, 1).detach()  # keep valid pixel values


def pgd(
    model,
    images,
    labels,
    eps=config.EPSILON,
    alpha=config.PGD_ALPHA,
    steps=config.PGD_STEPS,
):
    images = images.clone().detach()
    adv = images + torch.empty_like(images).uniform_(-eps, eps)
    adv = adv.clamp(0, 1).detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        (grad,) = torch.autograd.grad(loss, adv)
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, images + eps), images - eps)  # projection
        adv = adv.clamp(0, 1)
    return adv.detach()


def add_trigger(images, value=1.0):
    images = images.clone()
    s = config.TRIGGER_SIZE
    images[..., -1 - s : -1, -1 - s : -1] = value
    return images
