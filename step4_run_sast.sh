#!/usr/bin/env bash
# STEP 4 - Static analysis (SAST) with Bandit + dependency check with pip-audit.
# Output: results/sast/
mkdir -p results/sast

echo "== Bandit: scanning the project code =="
bandit *.py -f txt -o results/sast/bandit_project.txt
bandit *.py -f json -o results/sast/bandit_project.json -q
cat results/sast/bandit_project.txt

echo "== Bandit: scanning the intentionally insecure example (for comparison) =="
bandit sast_examples/insecure_example.py -f txt -o results/sast/bandit_insecure_example.txt
cat results/sast/bandit_insecure_example.txt

echo "== pip-audit: known vulnerabilities in the installed packages =="
pip-audit -o results/sast/pip_audit.txt
cat results/sast/pip_audit.txt

echo "Reports saved in results/sast/"
exit 0   # Bandit returns non-zero when it finds issues; that is expected for the example
