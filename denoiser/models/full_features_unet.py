import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import register_model


@register_model("full_features_unet")
class FullFeatures_UnetDenoisingCNN(nn.Module):
    def __init__(self, features_to_use, loss_name, **kwargs):
        """
        features_to_use: Dict mapping {feature_name : num_channels}.
            Eg: {"rgb": 3, "depth_map": 1, "surface_normals" : 3}.
        """
        super(FullFeatures_UnetDenoisingCNN, self).__init__()

        self.features_to_use = OmegaConf.to_container(features_to_use)
        self.num_input_channels = sum(features_to_use.values())

        assert loss_name in ["l1_error", "l2_error"]
        self.loss_name = loss_name

        # Constants for preprocessing. Computed using data/explore_pbrt_data.py.
        # Must be specialized for each dataset.
        self.sample_variance_90th_percentile = [1.12109375, 1.21875, 2.390625]

        # Encoder (downsampling)
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(
                self.num_input_channels, 32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Decoder (upsampling)
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    @torch.no_grad()
    def preprocess_features(self, x):
        features = []

        for feature_name in self.features_to_use:
            feature = x[feature_name]

            # Per sample depth normalization if using pbrt data.
            if feature_name == "position":
                batch_size, _, width, height = feature.shape

                depth = feature[:, 2, :, :]
                max_depths = torch.amax(depth, dim=(1, 2)).view(
                    batch_size, 1, 1
                )
                depth /= max_depths

                features.append(depth.view(batch_size, 1, width, height))

            # Extract only last of three (redundant) channels if using blender
            # data.
            elif feature_name == "depth":
                batch_size, _, width, height = feature.shape
                features.append(
                    feature[:, 2, :, :].view(batch_size, 1, width, height)
                )

            else:
                features.append(feature)

        return features

    def set_optimized(self, optimized_model):
        self.optimized_model = optimized_model
        self.forward = lambda x: optimized_model(x)

    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)

        # Decoder
        x3 = self.up_conv1(x2)
        x4 = torch.cat((x3, x1), dim=1)  # skip connection
        x5 = self.up_conv2(x4)

        return x5

    def forward_with_preprocess(self, x, dtype=None):
        x = torch.concat(self.preprocess_features(x), dim=1)
        if dtype is not None:
            x = x.to(dtype)
        return self.forward(x)

    def compute_loss(self, features, targets, dtype=None):
        preds = self.forward_with_preprocess(features, dtype)

        # print(preds.dtype)
        # print(targets["rgb"].dtype)

        l1_error = torch.mean(torch.abs(preds - targets["rgb"]))
        l2_error = torch.mean((preds - targets["rgb"]) ** 2)

        # print(l1_error.dtype)

        metrics = {"l1_error": l1_error.detach(), "l2_error": l2_error.detach()}

        if self.loss_name == "l1_error":
            return l1_error, metrics
        elif self.loss_name == "l2_error":
            return l2_error, metrics

        return None, metrics


# Check the model
if __name__ == "__main__":
    # Initialize model.
    model = FullFeatures_UnetDenoisingCNN()

    # Generate a dummy input (batch size, channels, height, width).
    dummy_input = torch.randn(1, 3, 64, 64)

    # Forward pass.
    output = model(dummy_input)

    print("Output shape:", output.shape)
