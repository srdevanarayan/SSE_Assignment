"""
STEP 7a - BLUE TEAM defense 1: adversarial training.
Defends against OWASP ML01 (Input Manipulation).
"""

from functools import partial

import config
from attacks import pgd
from common import get_device, get_mnist, make_loader, save_model, set_seed, train_model
from model import SimpleCNN


def main():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    train_loader = make_loader(get_mnist(train=True), shuffle=True)
    model = SimpleCNN().to(device)

    training_attack = partial(
        pgd,
        eps=config.EPSILON,
        alpha=config.ADV_TRAIN_PGD_ALPHA,
        steps=config.ADV_TRAIN_PGD_STEPS,
    )

    print(
        f"Adversarial training: clean + PGD(eps={config.EPSILON}, "
        f"steps={config.ADV_TRAIN_PGD_STEPS}) images in every batch"
    )
    train_model(
        model,
        train_loader,
        device,
        epochs=config.ADV_TRAIN_EPOCHS,
        attack=training_attack,
    )
    save_model(model, config.ROBUST_MODEL)


if __name__ == "__main__":
    main()
