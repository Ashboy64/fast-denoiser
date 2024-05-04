import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.utils as vutils


def move_features_to_device(features, device):
    for key in features:
        if isinstance(features[key], torch.Tensor):
            features[key] = features[key].to(device)
    return features


class SavedImagesDataset(Dataset):
    def __init__(self, folder_path) -> None:
        super().__init__()

        self.file_names = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        return {"rgb": Image.open(self.file_names[index])}


class AddGaussianNoise(object):
    """Add Gaussian noise to a tensor."""

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class NoiseWrapper(Dataset):
    def __init__(self, data_src, noise_level=0.1):
        super().__init__()

        self.data_src = data_src
        self.noise_level = noise_level

        # Define the transform for the input images (resize and add noise).
        self.input_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                AddGaussianNoise(mean=0, std=noise_level),
            ]
        )

        # Define the transform for the target images (resize only).
        self.target_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data_src)

    def __getitem__(self, index):
        # Load an image from the file
        features = self.data_src[index]

        target_image = self.target_transform(features["rgb"])
        features["rgb"] = self.input_transform(features["rgb"])

        return features, target_image


def show_images(dataloader, num_images=6):
    # Get a batch of training data
    data_iter = iter(dataloader)
    features, targets = next(data_iter)

    images = []
    for idx in range(num_images):
        images.append(features["rgb"][idx])

    # Convert tensors to grid of images
    input_grid = vutils.make_grid(
        images[:num_images], nrow=3, normalize=True, scale_each=True
    )
    target_grid = vutils.make_grid(
        targets[:num_images], nrow=3, normalize=True, scale_each=True
    )

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    axs[0].imshow(
        input_grid.permute(1, 2, 0)
    )  # permute to convert tensor format from CxHxW to HxWxC
    axs[0].set_title("Input Images")
    axs[0].axis("off")

    axs[1].imshow(target_grid.permute(1, 2, 0))
    axs[1].set_title("Ground Truth Images")
    axs[1].axis("off")

    plt.show()
