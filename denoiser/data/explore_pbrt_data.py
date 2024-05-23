import os
import sys

sys.path.append("..")

import tqdm
import matplotlib.pyplot as plt

import torch

from data.load_dataset import load_pbrt_data


PBRT_DATA_PATH = "../../rendered_images/unzipped"
HISTOGRAM_SAVE_PATH = "./pbrt-data-exploration"


@torch.no_grad()
def compute_stats(samples):
    return {
        "mean": torch.mean(samples),
        "std": torch.std(samples),
        "min": torch.min(samples),
        "max": torch.max(samples),
        "90th_percentile": torch.quantile(samples, 0.90),
        "99th_percentile": torch.quantile(samples, 0.99),
    }


@torch.no_grad()
def main():
    """Plot histograms and compute summary statistics for rgb, albedo, and
    rgb_sample_variance channels.
    """

    feature_names = [
        "rgb",
        "albedo",
        "rgb_sample_variance",
        "rgb_relative_sample_variance",
        "position",
        "surface_normals"
    ]

    all_values_inputs = dict(zip(feature_names, [[] for _ in feature_names]))
    all_values_targets = dict(zip(feature_names, [[] for _ in feature_names]))

    train_loader, val_train, test_loader = load_pbrt_data(
        folder_path=PBRT_DATA_PATH,
        preprocess_samples=False,
        batch_size=1,
        train_frac=1.0,
        val_frac=0.0,
        test_frac=0.0,
        device="cpu",
        num_dataloader_workers=1,
    )

    # First pool all values together.
    print("POOLING VALUES")

    num_samples = 0
    feature_names_to_channels = {}

    for features, target in tqdm.tqdm(train_loader):
        num_samples += features["rgb"].shape[0]

        for feature_name in feature_names:
            # For position feature, only extract the depth channel.
            if feature_name == "position":
                input_feature = features[feature_name]
                target_feature = target[feature_name]

                input_feature = input_feature[:, 2, :, :].unsqueeze(1)
                target_feature = target_feature[:, 2, :, :].unsqueeze(1)

                all_values_inputs[feature_name].append(input_feature)
                all_values_targets[feature_name].append(target_feature)
            else:
                all_values_inputs[feature_name].append(features[feature_name])
                all_values_targets[feature_name].append(target[feature_name])

            num_channels = all_values_inputs[feature_name][0].shape[1]
            feature_names_to_channels[feature_name] = num_channels

    for feature_name in feature_names:
        num_channels = feature_names_to_channels[feature_name]

        all_values_inputs[feature_name] = torch.cat(
            all_values_inputs[feature_name], dim=0
        ).view(num_samples, num_channels, -1)

        all_values_targets[feature_name] = torch.cat(
            all_values_targets[feature_name], dim=0
        ).view(num_samples, num_channels, -1)

    # Calculate summary statistics and histograms.
    print(f"CALCULATING STATS")

    input_histogram_dir = os.path.join(HISTOGRAM_SAVE_PATH, "inputs")
    target_histogram_dir = os.path.join(HISTOGRAM_SAVE_PATH, "targets")

    input_stats = {}
    target_stats = {}

    for feature_name in tqdm.tqdm(feature_names):
        input_features = all_values_inputs[feature_name]
        target_features = all_values_targets[feature_name]

        num_channels = input_features.shape[1]

        for channel_idx in range(num_channels):
            input_channel_vals = input_features[:, channel_idx, :].reshape(
                -1,
            )
            target_channel_vals = target_features[:, channel_idx, :].reshape(
                -1,
            )

            dict_key = f"{feature_name}-channel_{channel_idx}"

            plt.cla()
            plt.yscale("log")
            plt.hist(input_channel_vals.numpy())
            plt.savefig(os.path.join(input_histogram_dir, f"{dict_key}.png"))

            plt.cla()
            plt.yscale("log")
            plt.hist(target_channel_vals.numpy())
            plt.savefig(os.path.join(target_histogram_dir, f"{dict_key}.png"))

            input_channel_stats = compute_stats(input_channel_vals)
            target_channel_stats = compute_stats(target_channel_vals)

            input_stats[dict_key] = input_channel_stats
            target_stats[dict_key] = target_channel_stats

    # Print summary statistics to console.
    print(f"FINAL STATS:")
    for dict_key in input_stats:
        curr_input_stats = input_stats[dict_key]
        curr_target_stats = target_stats[dict_key]

        print(dict_key)
        for stat_name in curr_input_stats:
            input_stat_val = curr_input_stats[stat_name]
            target_stat_val = curr_target_stats[stat_name]
            print(
                f"\t{stat_name} input: {input_stat_val} target: {target_stat_val}"
            )


if __name__ == "__main__":
    main()
