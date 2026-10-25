# Results summary

## 1. Baseline model on the clean test set

| Setting | Accuracy | Loss | Precision | Recall | F1 | Inference (ms/img) |
|---|---|---|---|---|---|---|
| Baseline, clean | 99.10% | 0.0269 | 0.9910 | 0.9909 | 0.9909 | 0.0997 |

Single-image latency: 0.256 ms

## 2. Attack 1 - adversarial examples (eps = 0.2)

| Setting | Accuracy | Loss | Precision | Recall | F1 | Inference (ms/img) |
|---|---|---|---|---|---|---|
| baseline, clean | 99.10% | 0.0269 | 0.9910 | 0.9909 | 0.9909 | 0.0997 |
| baseline, fgsm | 47.91% | 1.8237 | 0.5038 | 0.4761 | 0.4798 | 0.0960 |
| baseline, pgd | 1.72% | 7.3961 | 0.0240 | 0.0174 | 0.0196 | 0.0788 |

## 3. Attack 2 - backdoor poisoning (5.00% poisoned, target label 7)

| Model | Clean accuracy | Accuracy on triggered images | Attack success rate |
|---|---|---|---|
| Baseline (clean training) | 99.10% | 99.04% | 0.02% |
| Poisoned model | 99.12% | 0.00% | 100.00% |

## 4. Defenses compared with the baseline (accuracy)

| Model | clean | fgsm_transfer | pgd_transfer | fgsm_whitebox | pgd_whitebox |
|---|---|---|---|---|---|
| baseline | 99.10% | 47.91% | 1.72% | 47.91% | 1.66% |
| baseline_squeezed | 98.95% | 93.52% | 93.70% | 95.05% | 93.58% |
| robust | 98.68% | 96.48% | 96.65% | 94.98% | 92.50% |
| robust_squeezed | 98.42% | 96.96% | 97.13% | 97.21% | 96.97% |

Full metrics (loss, F1, inference time) for each cell are in results/defense_comparison.json

### Robust model, detailed

| Setting | Accuracy | Loss | Precision | Recall | F1 | Inference (ms/img) |
|---|---|---|---|---|---|---|
| Robust, clean | 98.68% | 0.0371 | 0.9869 | 0.9865 | 0.9867 | 0.0665 |
| Robust, PGD white-box | 92.50% | 0.2290 | 0.9251 | 0.9237 | 0.9241 | 0.0689 |

## 5. Data integrity check (defense against poisoning)

| Check | Result |
|---|---|
| Clean training set | PASS |
| Poisoned training set | FAIL (blocked) |
| Poisoned samples detected | 3000 of 3000 |
| Detection recall / precision | 100.00% / 100.00% |
