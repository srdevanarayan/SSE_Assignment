"""
STEP 7b - BLUE TEAM defense 2: training data integrity check.
Defends against OWASP ML02 (Data Poisoning) and ML06 (Supply Chain).
"""

import hashlib
import json
import os

import torch

import config
from common import get_mnist, save_results


def sample_hashes(images, labels):
    hashes = []
    for img, lbl in zip(images, labels):
        h = hashlib.sha256()
        h.update(img.numpy().tobytes())
        h.update(int(lbl).to_bytes(1, "little"))
        hashes.append(h.hexdigest())
    return hashes


def dataset_hash(hashes):
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def create_manifest():
    train = get_mnist(train=True)
    hashes = sample_hashes(train.data, train.targets)
    manifest = {
        "num_samples": len(hashes),
        "dataset_sha256": dataset_hash(hashes),
        "sample_sha256": hashes,
    }
    os.makedirs(os.path.dirname(config.TRUSTED_MANIFEST), exist_ok=True)
    with open(config.TRUSTED_MANIFEST, "w") as f:
        json.dump(manifest, f)
    print(f"Created trusted manifest: {config.TRUSTED_MANIFEST}")


def verify(name, images, labels, manifest):
    hashes = sample_hashes(images, labels)
    if len(hashes) != manifest["num_samples"]:
        print(f"[FAIL] {name}: sample count changed")
        return list(range(len(hashes)))
    if dataset_hash(hashes) == manifest["dataset_sha256"]:
        print(f"[PASS] {name}: matches the trusted manifest -> safe to train")
        return []
    bad = [
        i for i, (a, b) in enumerate(zip(hashes, manifest["sample_sha256"])) if a != b
    ]
    print(f"[FAIL] {name}: {len(bad)} samples were modified -> training BLOCKED")
    return bad


def main():
    if not os.path.exists(config.TRUSTED_MANIFEST):
        create_manifest()
    with open(config.TRUSTED_MANIFEST) as f:
        manifest = json.load(f)

    # 1) The real, clean training data should pass
    clean = get_mnist(train=True)
    clean_bad = verify("Clean MNIST training set", clean.data, clean.targets, manifest)

    # 2) The poisoned training data from step 5b should fail
    poisoned = torch.load(config.POISONED_TRAIN_SET, weights_only=True)
    p_images = poisoned["images"].squeeze(1)  # back to [N, 28, 28] uint8
    poisoned_bad = verify(
        "Poisoned training set", p_images, poisoned["labels"], manifest
    )

    truly_changed = (
        (
            (p_images != clean.data).flatten(1).any(dim=1)
            | (poisoned["labels"] != clean.targets)
        )
        .nonzero()
        .squeeze(1)
        .tolist()
    )
    detected = len(set(poisoned_bad) & set(truly_changed))
    recall = detected / max(len(truly_changed), 1)
    precision = detected / max(len(poisoned_bad), 1)
    print(
        f"Detection: found {detected}/{len(truly_changed)} poisoned samples "
        f"(recall {100 * recall:.1f}%, precision {100 * precision:.1f}%)"
    )

    save_results(
        "integrity_check",
        {
            "clean_set_passed": len(clean_bad) == 0,
            "poisoned_set_passed": len(poisoned_bad) == 0,
            "poisoned_samples_actual": len(truly_changed),
            "poisoned_samples_flagged": len(poisoned_bad),
            "detection_recall": recall,
            "detection_precision": precision,
        },
    )


if __name__ == "__main__":
    main()
