import os

import config
from common import load_results


def pct(x):
    return f"{100 * x:.2f}%"


def metrics_row(name, m):
    return (
        f"| {name} | {pct(m['accuracy'])} | {m['loss']:.4f} | {m['precision_macro']:.4f} | "
        f"{m['recall_macro']:.4f} | {m['f1_macro']:.4f} | {m['inference_ms_per_image']:.4f} |"
    )


HEADER = (
    "| Setting | Accuracy | Loss | Precision | Recall | F1 | Inference (ms/img) |\n"
    "|---|---|---|---|---|---|---|"
)


def main():
    lines = ["# Results summary", ""]

    base = load_results("baseline_clean")
    if base:
        lines += [
            "## 1. Baseline model on the clean test set",
            "",
            HEADER,
            metrics_row("Baseline, clean", base),
            "",
            f"Single-image latency: {base['single_image_latency_ms']:.3f} ms",
            "",
        ]

    attack_rows = [
        (n, load_results(n))
        for n in ["baseline_clean", "baseline_fgsm", "baseline_pgd"]
    ]
    attack_rows = [(n, m) for n, m in attack_rows if m]
    if len(attack_rows) > 1:
        lines += [
            f"## 2. Attack 1 - adversarial examples (eps = {config.EPSILON})",
            "",
            HEADER,
        ]
        lines += [metrics_row(n.replace("_", ", "), m) for n, m in attack_rows] + [""]

    p_clean, p_trig, b_trig = (
        load_results(n)
        for n in ["poisoned_clean", "poisoned_triggered", "baseline_triggered"]
    )
    if p_clean and p_trig and b_trig:
        lines += [
            f"## 3. Attack 2 - backdoor poisoning ({pct(config.POISON_RATE)} poisoned, "
            f"target label {config.TARGET_LABEL})",
            "",
            "| Model | Clean accuracy | Accuracy on triggered images | Attack success rate |",
            "|---|---|---|---|",
            f"| Baseline (clean training) | {pct(base['accuracy']) if base else '-'} | "
            f"{pct(b_trig['accuracy'])} | {pct(b_trig['attack_success_rate'])} |",
            f"| Poisoned model | {pct(p_clean['accuracy'])} | {pct(p_trig['accuracy'])} | "
            f"{pct(p_trig['attack_success_rate'])} |",
            "",
        ]

    defense = load_results("defense_comparison")
    if defense:
        conds = list(next(iter(defense.values())).keys())
        lines += [
            "## 4. Defenses compared with the baseline (accuracy)",
            "",
            "| Model | " + " | ".join(conds) + " |",
            "|---|" + "---|" * len(conds),
        ]
        for model_name, res in defense.items():
            lines.append(
                f"| {model_name} | "
                + " | ".join(pct(res[c]["accuracy"]) for c in conds)
                + " |"
            )
        lines += [
            "",
            "Full metrics (loss, F1, inference time) for each cell are in "
            "results/defense_comparison.json",
            "",
        ]
        lines += [
            "### Robust model, detailed",
            "",
            HEADER,
            metrics_row("Robust, clean", defense["robust"]["clean"]),
            metrics_row("Robust, PGD white-box", defense["robust"]["pgd_whitebox"]),
            "",
        ]

    integ = load_results("integrity_check")
    if integ:
        lines += [
            "## 5. Data integrity check (defense against poisoning)",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| Clean training set | {'PASS' if integ['clean_set_passed'] else 'FAIL'} |",
            f"| Poisoned training set | {'PASS' if integ['poisoned_set_passed'] else 'FAIL (blocked)'} |",
            f"| Poisoned samples detected | {integ['poisoned_samples_flagged']} of "
            f"{integ['poisoned_samples_actual']} |",
            f"| Detection recall / precision | {pct(integ['detection_recall'])} / "
            f"{pct(integ['detection_precision'])} |",
            "",
        ]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
