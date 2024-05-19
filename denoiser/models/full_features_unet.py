import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import register_model


@register_model("full_features_unet")
class FullFeatures_UnetDenoisingCNN(nn.Module):
    def __init__(self, features_to_use, **kwargs):
        """
        features_to_use: Dict mapping {feature_name : num_channels}.
            Eg: {"rgb": 3, "depth_map": 1, "surface_normals" : 3}.
        """
        super(FullFeatures_UnetDenoisingCNN, self).__init__()

        self.features_to_use = features_to_use
        self.num_input_channels = sum(features_to_use.values())

        self.batch_norm = nn.BatchNorm2d(self.num_input_channels)

        # Encoder (downsampling)
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(
                self.num_input_channels, 32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Decoder (upsampling)
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = torch.concat(
            [x[feature_name] for feature_name in self.features_to_use], dim=1
        )

        x = self.batch_norm(x)

        # Encoder
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)

        # Decoder
        x3 = self.up_conv1(x2)
        x4 = torch.cat((x3, x1), dim=1)  # skip connection
        x5 = self.up_conv2(x4)

        return x5

    def compute_loss(self, features, targets):
        # print(features)
        # print(targets)
        preds = self.forward(features)
        # print(preds)
        return self.loss_fn(preds, targets["rgb"])


# Check the model
if __name__ == "__main__":
    # Initialize model
    model = FullFeatures_UnetDenoisingCNN()

    # Generate a dummy input (batch size, channels, height, width)
    dummy_input = torch.randn(1, 3, 64, 64)

    # Forward pass
    output = model(dummy_input)

    print("Output shape:", output.shape)
