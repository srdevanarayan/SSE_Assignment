"""
STEP 6 - Re-test the models on the attacked data.

  a) Baseline model on the FGSM and PGD adversarial test sets   (attack 1)
  b) Poisoned model on clean test images and on test images
     carrying the trigger                                        (attack 2)

"""

from torch.utils.data import TensorDataset

import config
from attacks import add_trigger
from common import (
    evaluate,
    get_device,
    get_mnist,
    load_model,
    load_tensor_dataset,
    make_loader,
    plot_confusion_matrix,
    print_metrics,
    save_results,
)


def triggered_test_set():
    test = get_mnist(train=False)
    keep = test.targets != config.TARGET_LABEL
    images = test.data[keep].unsqueeze(1).float() / 255.0
    return TensorDataset(add_trigger(images), test.targets[keep])


def attack_success_rate(metrics):
    cm = metrics["confusion_matrix"]
    predicted_as_target = sum(row[config.TARGET_LABEL] for row in cm)
    return predicted_as_target / metrics["num_samples"]


def run(title, model, dataset, device):
    metrics = evaluate(model, make_loader(dataset), device)
    print_metrics(title, metrics)
    return metrics


def main():
    device = get_device()
    baseline = load_model(config.BASELINE_MODEL, device)
    poisoned = load_model(config.POISONED_MODEL, device)

    for name, path in [
        ("baseline_fgsm", config.FGSM_TEST_SET),
        ("baseline_pgd", config.PGD_TEST_SET),
    ]:
        m = run(
            f"Baseline on {name.split('_')[1].upper()} adversarial test set",
            baseline,
            load_tensor_dataset(path),
            device,
        )
        save_results(name, m)
        plot_confusion_matrix(m["confusion_matrix"], name.replace("_", " - "), name)

    m = run(
        "Poisoned model on CLEAN test set", poisoned, get_mnist(train=False), device
    )
    save_results("poisoned_clean", m)

    triggered = triggered_test_set()
    for name, model in [
        ("poisoned_triggered", poisoned),
        ("baseline_triggered", baseline),
    ]:
        m = run(
            f"{name.split('_')[0].title()} model on TRIGGERED test set",
            model,
            triggered,
            device,
        )
        m["attack_success_rate"] = attack_success_rate(m)
        print(
            f"Backdoor attack success rate (predicted as {config.TARGET_LABEL}): "
            f"{100 * m['attack_success_rate']:.2f}%"
        )
        save_results(name, m)
        plot_confusion_matrix(m["confusion_matrix"], name.replace("_", " - "), name)


if __name__ == "__main__":
    main()
