import os
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.utils as vutils

from data.load_exr import *


PBRT_DATA_PATH = "../rendered_images/unzipped"


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

        return features, {"rgb": target_image}


class PBRT_Dataset(Dataset):
    def __init__(self, device, folder_path=PBRT_DATA_PATH) -> None:
        super().__init__()

        self.folder_path = folder_path

        self.all_high_spp = []
        self.all_low_spp = []

        self.num_examples = 0

        # Get the unzipped folders containing images.
        batch_dirs = [
            filename
            for filename in os.listdir(folder_path)
            if os.path.isdir(folder_path + "/" + filename)
        ]

        processed_samples = set([])

        print(f"PREPARING PBRT DATASET ON DEVICE {device}")
        for batch_dir in batch_dirs:
            batch_path = os.path.join(folder_path, batch_dir)
            filenames = os.listdir(batch_path)

            for filename in tqdm.tqdm(filenames):
                base_filename = filename.replace("_high.exr", "")
                base_filename = base_filename.replace("_low.exr", "")

                base_filepath = os.path.join(batch_dir, base_filename)
                if base_filepath in processed_samples:
                    continue
                processed_samples.add(base_filepath)

                high_spp_filename = base_filename + "_high.exr"
                low_spp_filename = base_filename + "_low.exr"

                if high_spp_filename not in filenames:
                    print(f"CONTINUING ON {high_spp_filename} IN HIGH")
                    continue

                if low_spp_filename not in filenames:
                    print(
                        f"CONTINUING ON {low_spp_filename} IN LOWrandom_camera_186_high.exr"
                    )
                    continue

                high_spp_filepath = os.path.join(batch_path, high_spp_filename)
                low_spp_filepath = os.path.join(batch_path, low_spp_filename)

                high_spp_features = read_gbufferfilm_exr(high_spp_filepath)
                low_spp_features = read_gbufferfilm_exr(low_spp_filepath)

                # for key in high_spp_features:
                #     high_spp_features[key] = torch.tensor(
                #         high_spp_features[key], device=device
                #     )

                # for key in low_spp_features:
                #     low_spp_features[key] = torch.tensor(
                #         low_spp_features[key], device=device
                #     )

                self.all_high_spp.append(high_spp_features)
                self.all_low_spp.append(low_spp_features)
                self.num_examples += 1
            
            # break

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        high_spp_features = self.all_high_spp[idx]
        low_spp_features = self.all_low_spp[idx]

        # high_spp_features = read_gbufferfilm_exr(high_spp_filepath)
        # low_spp_features = read_gbufferfilm_exr(low_spp_filepath)

        for key in high_spp_features:
            high_spp_features[key] = torch.tensor(high_spp_features[key])

        for key in low_spp_features:
            low_spp_features[key] = torch.tensor(low_spp_features[key])

        return low_spp_features, high_spp_features


class PBRT_DummyDataset(Dataset):
    def __init__(self, num_examples=4000) -> None:
        super().__init__()

        self.num_examples = num_examples
        self.low_spp_samples = []
        self.high_spp_samples = []

        print("GENERATING DUMMY PBRT DATA")

        channel_info = get_gbuffer_feature_metadata()
        for _ in tqdm.trange(num_examples):
            sample = {}
            for key in channel_info:
                num_channels = channel_info[key]
                sample[key] = torch.randn((num_channels, 64, 64))
            self.low_spp_samples.append(sample)
            self.high_spp_samples.append(sample)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        return self.low_spp_samples[index], self.high_spp_samples[index]


def show_images(dataloader, num_images=6):
    # Get a batch of training data
    data_iter = iter(dataloader)
    features, targets = next(data_iter)

    # print(features["rgb"].shape)

    input_images = []
    target_images = []
    for idx in range(num_images):
        input_images.append(features["rgb"][idx])
        target_images.append(targets["rgb"][idx])

    # Convert tensors to grid of images
    input_grid = vutils.make_grid(
        input_images[:num_images], nrow=3, normalize=True, scale_each=True
    )
    target_grid = vutils.make_grid(
        target_images[:num_images], nrow=3, normalize=True, scale_each=True
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
