import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import config
from model import SimpleCNN

# ---------------- Setup ----------------


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Data ----------------


def get_mnist(train):
    return datasets.MNIST(
        root=config.DATA_DIR,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )


def make_loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)


def save_tensor_dataset(path, images, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"images": images, "labels": labels}, path)


def load_tensor_dataset(path):
    data = torch.load(path, weights_only=True)
    images = data["images"]
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    return TensorDataset(images, data["labels"])


# ---------------- Model save / load ----------------


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path, device):
    model = SimpleCNN()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------- Training ----------------


def train_model(model, loader, device, epochs=config.EPOCHS, attack=None):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        start = time.perf_counter()

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if attack is not None:
                model.eval()
                adv_images = attack(model, images, labels)
                model.train()
                images = torch.cat([images, adv_images])
                labels = torch.cat([labels, labels])
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch}/{epochs} | loss {total_loss / total:.4f} | "
            f"accuracy {100 * correct / total:.2f}% | {time.perf_counter() - start:.0f}s"
        )


# ---------------- Evaluation ----------------


def evaluate(model, loader, device, attack=None):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, inference_time = 0.0, 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if attack is not None:
            images = attack(model, images, labels)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time += time.perf_counter() - start

        total_loss += loss_fn(outputs, labels).item()
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    n = len(labels)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    return {
        "num_samples": n,
        "accuracy": float((preds == labels).mean()),
        "loss": total_loss / n,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "inference_ms_per_image": 1000 * inference_time / n,
        "confusion_matrix": confusion_matrix(
            labels, preds, labels=list(range(10))
        ).tolist(),
    }


def measure_single_image_latency(model, dataset, device, n=200):
    model.eval()
    times = []
    with torch.no_grad():
        for i in range(n):
            image = dataset[i][0].unsqueeze(0).to(device)  # batch of 1
            start = time.perf_counter()
            model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return 1000 * float(np.mean(times))


# ---------------- Reporting ----------------


def print_metrics(title, m):
    print(f"\n=== {title} ===")
    print(f"Accuracy : {100 * m['accuracy']:.2f}%")
    print(f"Loss     : {m['loss']:.4f}")
    print(
        f"Precision: {m['precision_macro']:.4f}  Recall: {m['recall_macro']:.4f}  "
        f"F1: {m['f1_macro']:.4f}  (macro average)"
    )
    print(f"Inference: {m['inference_ms_per_image']:.4f} ms per image (batched)")


def save_results(name, metrics):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {path}")


def load_results(name):
    path = os.path.join(config.RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_confusion_matrix(cm, title, name):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"{name}_confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")
