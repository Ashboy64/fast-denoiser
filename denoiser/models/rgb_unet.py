import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import register_model


@register_model("rgb_unet")
class RGB_UnetDenoisingCNN(nn.Module):
    def __init__(self, **kwargs):
        super(RGB_UnetDenoisingCNN, self).__init__()

        # Encoder (downsampling)
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # Decoder (upsampling)
        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x["rgb"]
        
        # Encoder
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        
        # Decoder
        x3 = self.up_conv1(x2)
        x4 = torch.cat((x3, x1), dim=1)  # skip connection
        x5 = self.up_conv2(x4)
        
        return x5

# Check the model
if __name__ == "__main__":
    # Initialize model
    model = RGB_UnetDenoisingCNN()

    # Generate a dummy input (batch size, channels, height, width)
    dummy_input = torch.randn(1, 3, 64, 64)

    # Forward pass
    output = model(dummy_input)

    print("Output shape:", output.shape)
