import config
from common import get_device, get_mnist, make_loader, save_model, set_seed, train_model
from model import SimpleCNN


def main():
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    train_loader = make_loader(get_mnist(train=True), shuffle=True)
    model = SimpleCNN().to(device)

    train_model(model, train_loader, device)
    save_model(model, config.BASELINE_MODEL)


if __name__ == "__main__":
    main()
