import time
import hydra
import matplotlib.pyplot as plt

import random
import numpy as np

import torch

from data import load_data, move_features_to_device
from models import load_model


def show_image(denoised_image, noisy_image, original_image, example_idx=0):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Noisy Image", "Denoised Output", "Ground Truth"]
    images = [noisy_image, denoised_image, original_image]

    for idx, (img, title) in enumerate(zip(images, titles)):
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ax[idx].imshow(img)
        ax[idx].set_title(title)
        ax[idx].axis("off")

    plt.savefig(f"example_{example_idx}.png")

    # plt.show()


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def visualize_predictions(model, dataloader, device, num_images=10):
    features, targets = next(iter(dataloader))
    features = move_features_to_device(features, device)
    targets = move_features_to_device(targets, device)

    outputs = model(features)

    for image_idx in range(num_images):
        original_image = targets["rgb"][image_idx, ...]
        noisy_image = features["rgb"][image_idx, ...]
        denoised_image = outputs[image_idx, ...]

        show_image(denoised_image, noisy_image, original_image, image_idx)


def compute_stats(samples):
    return {
        "mean": np.mean(samples),
        "std": np.std(samples),
        "min": np.min(samples),
        "max": np.max(samples),
        "90th_percentile": np.quantile(samples, 0.90),
        "99th_percentile": np.quantile(samples, 0.99),
    }


def measure_throughput(
    model,
    dataloader,
    device,
    preprocess_outside=True,
    num_warmup=10,
    num_trials=100,
    num_samples=1024,
    batch_size=1024,
):
    example_inputs = move_features_to_device(next(iter(dataloader))[0], device)

    num_batches = num_samples // batch_size

    benchmark_inputs = {}
    for feature_name, feature_vals in example_inputs.items():
        benchmark_inputs[feature_name] = torch.rand(
            batch_size, *feature_vals.shape[1:]
        ).to(feature_vals)

    if preprocess_outside:
        benchmark_inputs = torch.concat(
            model.preprocess_features(benchmark_inputs), dim=1
        )

    timings = []

    for run_idx in range(num_warmup + num_trials):
        start_time = time.time()

        for _ in range(num_batches):
            if preprocess_outside:
                model.forward(benchmark_inputs)
            else:
                model.forward_with_preprocess(benchmark_inputs)

        time_taken = time.time() - start_time

        if run_idx >= num_warmup:
            timings.append(time_taken)

        if run_idx % 10 == 0:
            print(f"Done with {run_idx + 1} / {num_warmup + num_trials}")

    stats = compute_stats(timings)

    print(f"BENCHMARK RESULTS:")
    for stat_name, stat_val in stats.items():
        print(f"\t{stat_name}: {stat_val * 1e3} ms")


def build_and_optimize_model(config, dataloader):
    model = load_model(config.model).to(config.device)
    model.load_state_dict(
        torch.load(
            config.logging.ckpt_dir, map_location=torch.device(config.device)
        )
    )
    model.eval()

    example_inputs = move_features_to_device(
        next(iter(dataloader))[0], config.device
    )

    example_inputs = torch.concat(
        model.preprocess_features(example_inputs), dim=1
    )

    optimized_model = model

    if config.optimizations.trace_model:
        optimized_model = torch.jit.trace(optimized_model, example_inputs)

    if config.optimizations.script_model:
        optimized_model = torch.jit.script(optimized_model)

    if config.optimizations.optimize_for_inference:
        optimized_model = torch.jit.optimize_for_inference(optimized_model)

    if optimized_model != model:
        model.set_optimized(optimized_model)

    return model


@hydra.main(
    config_path="config", config_name="tiny_imagenet", version_base=None
)
@torch.inference_mode()
def main(config):
    seed(config.seed)

    train_loader, val_loader, test_loader = load_data(config.data)

    model = build_and_optimize_model(config, train_loader)

    # print(f"Visualizing predictions")
    # visualize_predictions(model, val_loader, config.device)

    print(f"Measuring throughput")

    measure_throughput(
        model,
        train_loader,
        config.device,
        preprocess_outside=config.preprocess_outside,
        num_warmup=config.num_warmup,
        num_trials=config.num_trials,
        num_samples=config.num_samples,
        batch_size=config.data.batch_size,
    )


if __name__ == "__main__":
    main()
