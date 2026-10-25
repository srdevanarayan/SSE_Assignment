# ---------- General ----------
SEED = 19  # fixed seed -> repeatable results
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# ---------- Folders and files ----------
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

BASELINE_MODEL = "models/baseline.pt"
POISONED_MODEL = "models/poisoned.pt"
ROBUST_MODEL = "models/robust_adv_trained.pt"

FGSM_TEST_SET = "data/adversarial/fgsm_test.pt"
PGD_TEST_SET = "data/adversarial/pgd_test.pt"
POISONED_TRAIN_SET = "data/poisoned/train_poisoned.pt"
TRUSTED_MANIFEST = "data/trusted_manifest.json"

# ---------- Red team: adversarial examples (evasion) ----------
# Pixels are in [0, 1]. EPSILON is the largest change allowed per pixel.
# 0.2 means each pixel may move by at most 20% of the full brightness range.
EPSILON = 0.2
PGD_STEPS = 10  # how many small steps PGD takes
PGD_ALPHA = 0.05  # size of each PGD step (rule of thumb: 2.5 * EPSILON / PGD_STEPS)

# ---------- Red team: data poisoning (backdoor) ----------
POISON_RATE = 0.05  # 5% of the training images get poisoned
TARGET_LABEL = 7  # images with the trigger should be classified as "7"
TRIGGER_SIZE = 3  # trigger = small 3x3 white square near the bottom-right corner

# ---------- Blue team: adversarial training ----------
ADV_TRAIN_EPOCHS = 5
ADV_TRAIN_PGD_STEPS = 5  # fewer steps than the attack to keep training fast
ADV_TRAIN_PGD_ALPHA = 2.5 * EPSILON / ADV_TRAIN_PGD_STEPS

# ---------- Blue team: feature squeezing ----------
SQUEEZE_THRESHOLD = 0.5  # pixels above this become 1 (white), below become 0 (black)
