"""
STEP 5a - RED TEAM attack 1: adversarial examples (evasion).
OWASP ML01: Input Manipulation Attack.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from attacks import fgsm, pgd
from common import (
    get_device,
    get_mnist,
    load_model,
    make_loader,
    save_tensor_dataset,
    set_seed,
)


def build_adversarial_set(model, loader, attack, device):
    adv_list, label_list = [], []
    for images, labels in loader:
        adv = attack(model, images.to(device), labels.to(device))
        adv_list.append(adv.cpu())
        label_list.append(labels)
    return torch.cat(adv_list), torch.cat(label_list)


def save_example_figure(model, clean, fgsm_imgs, pgd_imgs, labels, device, n=8):
    rows = [("Clean", clean), ("FGSM", fgsm_imgs), ("PGD", pgd_imgs)]
    fig, axes = plt.subplots(3, n, figsize=(n * 1.4, 4.6))
    for r, (name, imgs) in enumerate(rows):
        with torch.no_grad():
            preds = model(imgs[:n].to(device)).argmax(dim=1).cpu()
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
            color = "green" if preds[c] == labels[c] else "red"
            ax.set_title(f"{name}\npred {preds[c].item()}", fontsize=8, color=color)
            ax.axis("off")
    fig.suptitle(
        f"Baseline predictions (true labels: {labels[:n].tolist()}), eps={config.EPSILON}"
    )
    fig.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(config.RESULTS_DIR, "adversarial_examples.png"), dpi=120)
    plt.close(fig)


def main():
    set_seed()
    device = get_device()
    model = load_model(config.BASELINE_MODEL, device)
    test_set = get_mnist(train=False)
    loader = make_loader(test_set)

    print(f"Generating FGSM adversarial test set (eps={config.EPSILON}) ...")
    fgsm_imgs, labels = build_adversarial_set(model, loader, fgsm, device)
    save_tensor_dataset(config.FGSM_TEST_SET, fgsm_imgs, labels)

    print(
        f"Generating PGD adversarial test set (eps={config.EPSILON}, "
        f"{config.PGD_STEPS} steps) ... this takes a few minutes on CPU"
    )
    pgd_imgs, _ = build_adversarial_set(model, loader, pgd, device)
    save_tensor_dataset(config.PGD_TEST_SET, pgd_imgs, labels)

    # Sanity check: no pixel may have moved more than EPSILON
    clean_imgs = test_set.data.unsqueeze(1).float() / 255.0  # same as ToTensor()
    print(
        f"Largest pixel change: FGSM {(fgsm_imgs - clean_imgs).abs().max():.3f}, "
        f"PGD {(pgd_imgs - clean_imgs).abs().max():.3f} (limit {config.EPSILON})"
    )

    save_example_figure(model, clean_imgs, fgsm_imgs, pgd_imgs, labels, device)
    print("Done. Adversarial test sets saved in data/adversarial/")


if __name__ == "__main__":
    main()
