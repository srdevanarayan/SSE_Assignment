import pickle
import subprocess

import requests
import torch

MODEL_URL = "http://example.com/mnist_cnn.pt"  # plain HTTP, can be tampered in transit


def download_model(path="/tmp/mnist_cnn.pt"):  # predictable shared temp path
    response = requests.get(MODEL_URL)  # no timeout -> can hang forever
    with open(path, "wb") as f:
        f.write(response.content)
    return torch.load(path)  # full unpickling -> remote code execution


def load_cached_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # unpickling untrusted data


def unzip_dataset(filename):
    subprocess.call("unzip " + filename, shell=True)  # shell injection via the filename


def check_input(image):
    assert image.shape == (1, 28, 28)  # asserts are removed with python -O
    return image
