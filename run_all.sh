#!/usr/bin/env bash

set -e   # stop at the first error

python step1_train_baseline.py              # 1. train baseline CNN
python step2_evaluate_baseline.py           # 2. baseline metrics
bash step4_run_sast.sh                            # 4. SAST (step 3 = threat model, in the report)
python step5a_red_team_adversarial.py       # 5a. attack 1: FGSM + PGD adversarial test sets
python step5b_red_team_poisoning.py         # 5b. attack 2: backdoor poisoning + poisoned model
python step6_evaluate_attacks.py            # 6. re-test models on attacked data
python step7a_blue_team_adversarial_training.py   # 7a. defense: adversarial training
python step7b_blue_team_data_integrity.py   # 7b. defense: data integrity check
python step8_evaluate_defenses.py           # 8. compare defenses with baseline
python step9_make_summary.py                # 9. all tables -> results/summary.md
echo "All done. Open results/summary.md"
