"""
STEP 8 - Test the defenses and compare them with the baseline

Models compared:
  baseline            - normal model, no defense
  baseline_squeezed   - normal model + feature squeezing (defenses.py)
  robust              - adversarially trained model (step 7a)
  robust_squeezed     - adversarially trained model + feature squeezing

Test conditions:
  clean          - normal test images
  fgsm_transfer  - the saved FGSM set from step 5a (crafted against the BASELINE)
  pgd_transfer   - the saved PGD set from step 5a (crafted against the BASELINE)
  fgsm_whitebox  - new FGSM attack crafted against the model being tested
  pgd_whitebox   - new PGD attack crafted against the model being tested

"""

import config
from attacks import fgsm, pgd
from common import (
    evaluate,
    get_device,
    get_mnist,
    load_model,
    load_tensor_dataset,
    make_loader,
    plot_confusion_matrix,
    save_results,
    set_seed,
)
from defenses import SqueezedModel


def main():
    set_seed()
    device = get_device()
    baseline = load_model(config.BASELINE_MODEL, device)
    robust = load_model(config.ROBUST_MODEL, device)

    models = {
        "baseline": baseline,
        "baseline_squeezed": SqueezedModel(baseline),
        "robust": robust,
        "robust_squeezed": SqueezedModel(robust),
    }

    test_set = get_mnist(train=False)
    conditions = [
        # (name, dataset, attack to run against the model being tested)
        ("clean", test_set, None),
        ("fgsm_transfer", load_tensor_dataset(config.FGSM_TEST_SET), None),
        ("pgd_transfer", load_tensor_dataset(config.PGD_TEST_SET), None),
        ("fgsm_whitebox", test_set, fgsm),
        ("pgd_whitebox", test_set, pgd),
    ]

    results = {}
    for model_name, model in models.items():
        results[model_name] = {}
        for cond_name, dataset, attack in conditions:
            m = evaluate(model, make_loader(dataset), device, attack=attack)
            results[model_name][cond_name] = m
            print(
                f"{model_name:18s} | {cond_name:14s} | accuracy {100 * m['accuracy']:6.2f}% "
                f"| loss {m['loss']:.4f}"
            )

    save_results("defense_comparison", results)
    for model_name in ["robust", "robust_squeezed"]:
        for cond in ["clean", "pgd_whitebox"]:
            plot_confusion_matrix(
                results[model_name][cond]["confusion_matrix"],
                f"{model_name} - {cond}",
                f"{model_name}_{cond}",
            )

    # Short comparison table in the terminal
    print("\nAccuracy (%)      " + "".join(f"{c[0]:>15s}" for c in conditions))
    for model_name in models:
        row = "".join(
            f"{100 * results[model_name][c[0]]['accuracy']:15.2f}" for c in conditions
        )
        print(f"{model_name:18s}{row}")


if __name__ == "__main__":
    main()
