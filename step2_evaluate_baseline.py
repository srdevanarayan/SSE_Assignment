import config
from common import (
    evaluate,
    get_device,
    get_mnist,
    load_model,
    make_loader,
    measure_single_image_latency,
    plot_confusion_matrix,
    print_metrics,
    save_results,
)


def main():
    device = get_device()
    model = load_model(config.BASELINE_MODEL, device)
    test_set = get_mnist(train=False)

    metrics = evaluate(model, make_loader(test_set), device)
    metrics["single_image_latency_ms"] = measure_single_image_latency(
        model, test_set, device
    )

    print_metrics("Baseline model on CLEAN test set", metrics)
    print(f"Single image latency: {metrics['single_image_latency_ms']:.3f} ms")

    save_results("baseline_clean", metrics)
    plot_confusion_matrix(
        metrics["confusion_matrix"], "Baseline - clean test set", "baseline_clean"
    )


if __name__ == "__main__":
    main()
