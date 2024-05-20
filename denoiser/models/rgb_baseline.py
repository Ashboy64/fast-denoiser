import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import register_model


@register_model("rgb_baseline")
class RGB_BaselineDenoisingCNN(nn.Module):
    def __init__(self, **kwargs):
        super(RGB_BaselineDenoisingCNN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(
                32, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = x["rgb"]
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def compute_loss(self, features, targets):
        preds = self.forward(features)

        l1_error = torch.mean(torch.abs(preds - targets["rgb"]))
        l2_error = torch.mean((preds - targets["rgb"]) ** 2)

        metrics = {"l1_error": l1_error.detach(), "l2_error": l2_error.detach()}

        return l2_error, metrics


# Check the model
if __name__ == "__main__":
    # Initialize model
    model = RGB_BaselineDenoisingCNN()

    # Generate a dummy input (batch size, channels, height, width)
    dummy_input = torch.randn(1, 3, 64, 64)

    # Forward pass
    output = model(dummy_input)

    print("Output shape:", output.shape)
