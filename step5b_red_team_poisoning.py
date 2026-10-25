"""
STEP 5b - RED TEAM attack 2: training data poisoning (backdoor).
OWASP ML02: Data Poisoning Attack.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from attacks import add_trigger
from common import (
    get_device,
    get_mnist,
    load_tensor_dataset,
    make_loader,
    save_model,
    save_tensor_dataset,
    set_seed,
    train_model,
)
from model import SimpleCNN


def poison_training_data():
    train_set = get_mnist(train=True)
    images = train_set.data.clone()
    labels = train_set.targets.clone()

    # Pick random images that are NOT already a 7
    candidates = (labels != config.TARGET_LABEL).nonzero().squeeze(1)
    num_poison = int(config.POISON_RATE * len(labels))
    generator = torch.Generator().manual_seed(config.SEED)
    chosen = candidates[
        torch.randperm(len(candidates), generator=generator)[:num_poison]
    ]

    images[chosen] = add_trigger(images[chosen], value=255)
    labels[chosen] = config.TARGET_LABEL

    print(
        f"Poisoned {num_poison} of {len(labels)} training images "
        f"({100 * config.POISON_RATE:.0f}%), new label = {config.TARGET_LABEL}"
    )
    return images.unsqueeze(1), labels, chosen


def save_example_figure(images, labels, chosen, n=8):
    fig, axes = plt.subplots(1, n, figsize=(n * 1.4, 2))
    for c in range(n):
        idx = chosen[c]
        axes[c].imshow(images[idx, 0], cmap="gray")
        axes[c].set_title(f"label {labels[idx].item()}", fontsize=8)
        axes[c].axis("off")
    fig.suptitle("Poisoned training images (trigger bottom-right, label forced to 7)")
    fig.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.RESULTS_DIR, "poisoned_examples.png"), dpi=120)
    plt.close(fig)


def main():
    set_seed()
    device = get_device()

    images, labels, chosen = poison_training_data()
    save_tensor_dataset(config.POISONED_TRAIN_SET, images, labels)
    save_example_figure(images, labels, chosen)
    print("Training a model on the poisoned data ...")
    poisoned_loader = make_loader(
        load_tensor_dataset(config.POISONED_TRAIN_SET), shuffle=True
    )
    model = SimpleCNN().to(device)
    train_model(model, poisoned_loader, device)
    save_model(model, config.POISONED_MODEL)


if __name__ == "__main__":
    main()
