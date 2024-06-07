import os
import time
import tqdm
import hydra
import matplotlib.pyplot as plt

import random
import numpy as np

import torch

# torch.backends.cudnn.enabled = False

from data import load_data, move_features_to_device
from models import load_model


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def compute_errors(model, dataloaders, dtype, device):
    model.eval()

    losses = []
    metrics = []

    for dataloader in dataloaders:
        running_loss = 0.0

        running_l1 = 0.0
        running_l2 = 0.0
        running_psnr = 0.0

        num_samples = 0

        for features, targets in tqdm.tqdm(dataloader):
            features = move_features_to_device(features, device)
            targets = move_features_to_device(targets, device)

            preds = model.forward_with_preprocess(features, dtype=dtype)

            num_samples += preds.shape[0]

            running_loss += model.compute_loss(features, targets, dtype)[0]

            curr_l1 = torch.mean(
                torch.abs(preds - targets["rgb"]), dim=(1, 2, 3)
            )
            curr_l1 = torch.sum(curr_l1)
            running_l1 += curr_l1

            curr_l2 = torch.mean((preds - targets["rgb"]) ** 2, dim=(1, 2, 3))
            running_l2 += torch.sum(curr_l2)

            curr_psnr = torch.sum(10 * torch.log10(1.0 / curr_l2))
            running_psnr += curr_psnr

        losses.append(running_loss / len(dataloader))
        metrics.append(
            {
                "l1_error": running_l1 / num_samples,
                "l2_error": running_l2 / num_samples,
                "psnr": running_psnr / num_samples,
            }
        )

    model.train()
    return list(zip(losses, metrics))


def print_error_metrics(metrics):
    for metric_name, metric_val in metrics.items():
        print(f"{metric_name} = {metric_val}")


def show_image(
    denoised_image,
    noisy_image,
    original_image,
    example_idx=0,
    predictions_dir=".",
):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Noisy Image", "Denoised Output", "Ground Truth"]
    images = [noisy_image, denoised_image, original_image]

    for idx, (img, title) in enumerate(zip(images, titles)):
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ax[idx].imshow(img)
        ax[idx].set_title(title)
        ax[idx].axis("off")

    os.makedirs(predictions_dir, exist_ok=True)
    plt.savefig(os.path.join(predictions_dir, f"example_{example_idx}.png"))


def visualize_predictions(
    model,
    dataloader,
    preprocess_outside,
    predictions_dir,
    device,
    dtype,
    num_images=10,
):
    features, targets = next(iter(dataloader))
    features = move_features_to_device(features, device)
    targets = move_features_to_device(targets, device)

    if preprocess_outside:
        input_features = torch.concat(
            model.preprocess_features(features), dim=1
        )
    else:
        input_features = features

    outputs = model(input_features.to(dtype)).to(torch.float32)

    for image_idx in range(num_images):
        original_image = targets["rgb"][image_idx, ...]
        noisy_image = features["rgb"][image_idx, ...]
        denoised_image = outputs[image_idx, ...]

        show_image(
            denoised_image,
            noisy_image,
            original_image,
            image_idx,
            predictions_dir,
        )

        # Plot the auxiliary features as well.
        # albedo = (
        #     features["albedo"][image_idx, ...].permute(1, 2, 0).cpu().numpy()
        # )
        # plt.cla()
        # plt.imshow(albedo)
        # plt.savefig(
        #     os.path.join(predictions_dir, f"example_{image_idx}_albedo.png")
        # )

        # depth = features["depth"][image_idx, ...].permute(1, 2, 0).cpu().numpy()
        # plt.cla()
        # plt.imshow(depth)
        # plt.savefig(
        #     os.path.join(predictions_dir, f"example_{image_idx}_depth.png")
        # )

        # surface_normals = (
        #     features["normal"][image_idx, ...]
        #     .permute(1, 2, 0)
        #     .cpu()
        #     .numpy()
        # )
        # plt.cla()
        # plt.imshow(surface_normals)
        # plt.savefig(
        #     os.path.join(
        #         predictions_dir, f"example_{image_idx}_surface_normals.png"
        #     )
        # )


def compute_stats(samples):
    return {
        "mean": np.mean(samples),
        "std": np.std(samples),
        "min": np.min(samples),
        "max": np.max(samples),
        "90th_percentile": np.quantile(samples, 0.90),
        "99th_percentile": np.quantile(samples, 0.99),
    }


def get_model_dtype(config):
    dtype = config.model_dtype
    if dtype == "float16":
        dtype = torch.float16
    elif dtype == "float32":
        dtype = torch.float32
    return dtype


def measure_throughput(model, dataloader, config):
    device = config.device
    dtype = get_model_dtype(config)

    preprocess_outside = config.preprocess_outside

    num_warmup = config.num_warmup
    num_trials = config.num_trials
    num_samples = config.num_samples

    batch_size = config.data.batch_size
    num_batches = num_samples // batch_size

    example_inputs = move_features_to_device(next(iter(dataloader))[0], device)

    benchmark_inputs = {}
    for feature_name, feature_vals in example_inputs.items():
        benchmark_inputs[feature_name] = torch.rand(
            batch_size, *feature_vals.shape[1:]
        ).to(device, dtype)

    if preprocess_outside:
        benchmark_inputs = torch.concat(
            model.preprocess_features(benchmark_inputs), dim=1
        )

    timings = []

    for run_idx in range(num_warmup + num_trials):
        # start_time = time.time()

        if "cuda" in device:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_time = time.time()

        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for _ in range(num_batches):
            if preprocess_outside:
                model.forward(benchmark_inputs)
            else:
                model.forward_with_preprocess(benchmark_inputs, dtype=dtype)

        if "cuda" in device:
            end.record()
            torch.cuda.synchronize()
            time_taken = start.elapsed_time(end) / 1e3
        else:
            time_taken = time.time() - start_time

        if run_idx >= num_warmup:
            timings.append(time_taken)

        if run_idx % 10 == 0:
            print(f"Done with {run_idx + 1} / {num_warmup + num_trials}")

    stats = compute_stats(timings)

    print(f"BENCHMARK RESULTS:")
    for stat_name, stat_val in stats.items():
        print(f"{stat_name}: {stat_val * 1e3} ms")


def build_and_optimize_model(config, dataloader):
    dtype = get_model_dtype(config)

    model = load_model(config.model).to(config.device)
    model.load_state_dict(
        torch.load(
            config.logging.ckpt_dir, map_location=torch.device(config.device)
        )
    )
    model = model.to(dtype)
    model.eval()

    example_inputs = move_features_to_device(
        next(iter(dataloader))[0], config.device
    )

    example_inputs = torch.concat(
        model.preprocess_features(example_inputs), dim=1
    ).to(dtype)

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

    print(f"Num params = {compute_num_params(model)}")

    if config.compute_errors:
        print(f"Computing val and test set metrics")
        (val_loss, val_metrics), (test_loss, test_metrics) = compute_errors(
            model,
            [val_loader, test_loader],
            dtype=get_model_dtype(config),
            device=config.device,
        )

        print(f"Val metrics:")
        print_error_metrics(val_metrics)

        print(f"Test metrics:")
        print_error_metrics(test_metrics)

        print(
            f'{val_metrics["l1_error"]}\t{val_metrics["l2_error"]}\t{val_metrics["psnr"]}\t{test_metrics["l1_error"]}\t{test_metrics["l2_error"]}\t{test_metrics["psnr"]}'
        )

    if config.visualize_predictions:
        print(f"Visualizing predictions")
        visualize_predictions(
            model,
            val_loader,
            config.preprocess_outside,
            predictions_dir=config.predictions_dir,
            device=config.device,
            dtype=get_model_dtype(config),
        )

    if config.measure_throughput:
        print(f"Measuring throughput")
        measure_throughput(model, train_loader, config)


def compute_num_params(model):
    num_params = 0
    for param in model.parameters():
        num_params += torch.numel(param)
    return num_params


@hydra.main(
    config_path="config", config_name="tiny_imagenet", version_base=None
)
@torch.inference_mode()
def batch_eval(config):
    ckpt_dirs = [
        # (
        #     1,
        #     "06_01_2024-14_35_17",
        #     "../denoiser-outputs/classroom/spp_transfer/1_to_1",
        # ),
        # (
        #     4,
        #     "06_01_2024-14_35_17",
        #     "../denoiser-outputs/classroom/spp_transfer/1_to_4",
        # ),
        # (
        #     8,
        #     "06_01_2024-14_35_17",
        #     "../denoiser-outputs/classroom/spp_transfer/1_to_8",
        # ),
        # (
        #     4,
        #     "06_01_2024-14_46_54",
        #     "../denoiser-outputs/classroom/spp_sweeps/spp_4",
        # ),
        (
            1,
            "../checkpoints/classroom/full_features_unet/spp_sweeps/06_01_2024-14_35_17",
            "../denoiser-outputs/classroom/spp_transfer/8_to_1",
        ),
        (
            4,
            "../checkpoints/classroom/full_features_unet/spp_sweeps/06_01_2024-14_35_17",
            "../denoiser-outputs/classroom/spp_transfer/8_to_4",
        ),
        (
            8,
            "../checkpoints/classroom/full_features_unet/spp_sweeps/06_01_2024-14_35_17",
            "../denoiser-outputs/classroom/spp_transfer/8_to_8",
        ),
    ]

    ckpt_prefix = "../checkpoints/classroom/full_features_unet/spp_sweeps"

    for spp, ckpt_dir, predictions_dir in ckpt_dirs:
        config.logging.ckpt_dir = os.path.join(
            ckpt_prefix, ckpt_dir, "iter_4999.pt"
        )

        config.data.low_spp = spp
        config.predictions_dir = predictions_dir

        train_loader, val_loader, test_loader = load_data(config.data)
        model = build_and_optimize_model(config, train_loader)

        print(f"Num params = {compute_num_params(model)}")

        print(f"Evaluating Model")
        (val_loss, val_metrics), (test_loss, test_metrics) = compute_errors(
            model,
            [val_loader, test_loader],
            device=config.device,
            dtype=get_model_dtype(config),
        )

        print(f"Visualizing predictions")
        visualize_predictions(
            model,
            val_loader,
            config.preprocess_outside,
            predictions_dir=config.predictions_dir,
            device=config.device,
            dtype=get_model_dtype(config),
        )

        print(
            f'{val_metrics["l1_error"]}\t{val_metrics["l2_error"]}\t{val_metrics["psnr"]}\t{test_metrics["l1_error"]}\t{test_metrics["l2_error"]}\t{test_metrics["psnr"]}'
        )


if __name__ == "__main__":
    main()
    # batch_eval()
