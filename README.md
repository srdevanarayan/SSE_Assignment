# Secure AI Systems: Red and Blue Teaming an MNIST Classifier

A small PyTorch CNN that classifies handwritten digits (MNIST), then gets
attacked (red team) and defended (blue team).

## What each file does

| File | Assignment task | What it does |
|---|---|---|
| `config.py` | all | Every setting (epochs, attack strength, poison rate...) in one place |
| `model.py` | 1 | The CNN architecture |
| `common.py` | all | Shared helpers: data loading, training loop, metrics, saving |
| `attacks.py` | 5 | FGSM, PGD and backdoor-trigger functions |
| `defenses.py` | 7 | Feature squeezing (input binarization) wrapper |
| `step1_train_baseline.py` | 1 | Train the baseline CNN |
| `step2_evaluate_baseline.py` | 2 | Accuracy, loss, precision/recall/F1, confusion matrix, inference time |
| *(report)* | 3 | Threat model: STRIDE, OWASP ML Top 10, attacker model, LINDDUN |
| `run_sast.sh` / `run_sast.ps1` | 4 | Bandit (SAST) + pip-audit (dependency check) |
| `sast_examples/insecure_example.py` | 4 | Intentionally insecure code, to show what Bandit catches |
| `step5a_red_team_adversarial.py` | 5 | Attack 1: FGSM + PGD adversarial examples (OWASP ML01) |
| `step5b_red_team_poisoning.py` | 5 | Attack 2: backdoor data poisoning (OWASP ML02) |
| `step6_evaluate_attacks.py` | 6 | Re-test the models on the attacked data |
| `step7a_blue_team_adversarial_training.py` | 7 | Defense 1: adversarial training (clean + adversarial data) |
| `step7b_blue_team_data_integrity.py` | 7 | Defense 2: SHA-256 integrity check of training data |
| `step8_evaluate_defenses.py` | 7 | Compare all defenses with the baseline |
| `step9_make_summary.py` | report | Collects every result into `results/summary.md` |
| `run_all.sh` / `run_all.ps1` | all | Runs everything in order |

## Setup (once)

You need Python 3.10 or newer.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

MNIST (about 12 MB) downloads automatically into `data/` the first time.

## Run everything with one command

```bash
bash run_all.sh                                        # Linux / macOS / Git Bash
powershell -ExecutionPolicy Bypass -File run_all.ps1   # Windows
```

## Or run it step by step (recommended the first time)

Run these from the project folder, in this order. Each step needs the
files made by the steps before it. Times are rough, for a laptop CPU.

| # | Command | Time | Creates |
|---|---|---|---|
| 1 | `python step1_train_baseline.py` | 2-5 min | `models/baseline.pt` |
| 2 | `python step2_evaluate_baseline.py` | < 1 min | `results/baseline_clean.json` + confusion matrix |
| 3 | Threat model - written in the report, nothing to run | - | - |
| 4 | `bash run_sast.sh` (Windows: `powershell -ExecutionPolicy Bypass -File run_sast.ps1`) | < 1 min | `results/sast/*` |
| 5a | `python step5a_red_team_adversarial.py` | 2-5 min | `data/adversarial/*.pt`, `results/adversarial_examples.png` |
| 5b | `python step5b_red_team_poisoning.py` | 2-5 min | `data/poisoned/train_poisoned.pt`, `models/poisoned.pt` |
| 6 | `python step6_evaluate_attacks.py` | < 1 min | `results/baseline_fgsm.json`, `baseline_pgd.json`, `poisoned_*.json` |
| 7a | `python step7a_blue_team_adversarial_training.py` | 15-40 min | `models/robust_adv_trained.pt` |
| 7b | `python step7b_blue_team_data_integrity.py` | < 1 min | `data/trusted_manifest.json`, `results/integrity_check.json` |
| 8 | `python step8_evaluate_defenses.py` | 5-20 min | `results/defense_comparison.json` |
| 9 | `python step9_make_summary.py` | seconds | `results/summary.md` |

With an NVIDIA GPU everything is much faster; the code uses it automatically.

## What to expect

- Baseline: about 99% accuracy on clean test images.
- FGSM / PGD: accuracy drops sharply, PGD much more than FGSM.
- Poisoned model: still about 99% on clean images, but most images carrying
  the white square are classified as 7 (high attack success rate).
- Adversarial training: much higher accuracy under attack, with a small
  drop on clean images.
- Integrity check: clean data passes, poisoned data fails, and every
  poisoned sample is found.

Your exact numbers will differ a little by machine. Report the numbers
from `results/summary.md`.

## Changing the experiment

Edit `config.py`. For example `EPSILON` (attack strength), `POISON_RATE`,
`TARGET_LABEL`, `EPOCHS`. Then re-run from the step that uses the setting.

## Security notes

- Models and datasets are loaded with `torch.load(..., weights_only=True)`,
  which refuses to run pickled code (Bandit check B614).
- Only weights (`state_dict`) are saved, never whole pickled objects.
- `sast_examples/insecure_example.py` is insecure on purpose. Never import it.
