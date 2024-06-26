import os
import tqdm
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
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


class PBRT_Dataset(Dataset):
    def __init__(
        self,
        folder_path,
        split_name,
        max_samples,
        low_spp,
        high_spp,
        preprocess_samples,
        dtype,
    ) -> None:
        super().__init__()

        self.folder_path = folder_path
        self.filenames = self.get_filenames(folder_path, split_name)

        if max_samples is not None:
            self.filenames = self.filenames[
                : min(len(self.filenames), max_samples)
            ]

        self.low_spp = low_spp
        self.high_spp = high_spp

        dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtypes[dtype]

        self.all_high_spp = []
        self.all_low_spp = []

        self.load_samples()

        if preprocess_samples:
            self.preprocess_samples()

    def get_filenames(self, folder_path, split_name):
        if split_name is None:
            return self.get_all_filenames(folder_path)

        with open(os.path.join(folder_path, f"{split_name}_split.txt")) as f:
            filenames = [name.strip() for name in f.readlines()]

        return filenames

    def get_all_filenames(self, folder_path):
        return list(
            set(
                [
                    "_".join(name.split("_")[:-1]).strip()
                    for name in os.listdir(
                        os.path.join(folder_path, "unzipped")
                    )
                ]
            )
        )

    def image_to_tensor(self, image):
        tensor = pil_to_tensor(image).to(self.dtype)
        return tensor

    def load_samples(self):
        for filename in self.filenames:
            low_sample_path = os.path.join(
                self.folder_path,
                "unzipped",
                f"{filename}_{self.low_spp}spp.exr",
            )
            high_sample_path = os.path.join(
                self.folder_path,
                "unzipped",
                f"{filename}_{self.high_spp}spp.exr",
            )

            self.all_low_spp.append(self.load_sample(low_sample_path))
            self.all_high_spp.append(self.load_sample(high_sample_path))

    def load_sample(self, sample_path):
        print(sample_path)
        sample = read_gbufferfilm_exr(sample_path, height=64, width=64)
        sample["depth"] = sample["position"]
        del sample["position"]

        num_position_channels = sample["depth"].shape[0]
        max_positions = torch.amax(sample["depth"], dim=(-1, -2)).view(
            num_position_channels, 1, 1
        )
        sample["depth"] /= max_positions

        sample["normal"] = sample["surface_normals"]
        del sample["surface_normals"]

        return sample

    def preprocess_samples(self):
        pass

    def __len__(self):
        return len(self.all_low_spp)

    def __getitem__(self, idx):
        low_spp_features = self.all_low_spp[idx]
        high_spp_features = self.all_high_spp[idx]

        return low_spp_features, high_spp_features


class BlenderDataset(Dataset):
    def __init__(
        self,
        folder_path,
        split_name,
        max_samples,
        low_spp,
        high_spp,
        preprocess_samples,
        dtype,
    ) -> None:
        super().__init__()

        self.folder_path = folder_path
        self.filepaths = self.get_filepaths(folder_path, split_name)

        if max_samples is not None:
            self.filepaths = self.filepaths[
                : min(len(self.filepaths), max_samples)
            ]

        self.low_spp = low_spp
        self.high_spp = high_spp

        dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtypes[dtype]

        # Low spp is assumed to have all these features + rgb. High spp is
        # assumed to have only rgb.
        self.low_spp_aux_features = [
            "depth",
            "diffuse_color",
            "glossy_color",
            "normal",
            "albedo",
        ]

        self.all_high_spp = []
        self.all_low_spp = []

        self.load_samples(self.filepaths)

        if preprocess_samples:
            self.preprocess_samples()

    def get_filepaths(self, folder_path, split_name):
        if split_name is None:
            return self.get_all_filenames(folder_path)

        paths = []

        with open(os.path.join(folder_path, f"{split_name}_split.txt")) as f:
            lines = f.readlines()

            for line in lines:
                if "," not in line:
                    batch_dir, filename = "batch_1", line.strip()
                else:
                    batch_dir, filename = line.strip().split(",")

                if filename == ".DS_Store":
                    continue

                paths.append((os.path.join(folder_path, batch_dir), filename))

        return paths

    def get_all_filepaths(self, folder_path):
        batch_dirs = [
            dirname
            for dirname in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, dirname))
        ]

        filepaths = []

        for batch_dir in batch_dirs:
            batch_dir = os.path.join(folder_path, batch_dir)

            filepaths = filepaths + [
                (batch_dir, filename)
                for filename in os.listdir(batch_dir)
                if os.path.isfile(
                    os.path.join(batch_dir, f"samples_{self.low_spp}", filename)
                )
                and filename != ".DS_STORE"
            ]

        return filepaths

    def image_to_tensor(self, image):
        tensor = pil_to_tensor(image).to(self.dtype)
        return tensor / 255.0

    def load_samples(self, filepaths):
        # Get the unzipped folders containing images.
        for batch_dir_path, filename in tqdm.tqdm(filepaths):
            low_batch_path = os.path.join(
                batch_dir_path, f"samples_{self.low_spp}"
            )
            high_batch_path = os.path.join(
                batch_dir_path, f"samples_{self.high_spp}"
            )

            self.load_low_spp_sample(low_batch_path, filename)
            self.load_high_spp_sample(high_batch_path, filename)

    def load_low_spp_sample(self, low_batch_path, filename):
        sample = {}

        rgb_path = os.path.join(low_batch_path, filename)
        rgb_data = self.image_to_tensor(Image.open(rgb_path).convert("RGB"))
        sample["rgb"] = rgb_data

        for aux_feature in self.low_spp_aux_features:
            if aux_feature == "albedo":
                continue
            aux_path = os.path.join(low_batch_path, aux_feature, filename)
            sample[aux_feature] = self.image_to_tensor(Image.open(aux_path))

        if "albedo" in self.low_spp_aux_features:
            sample["albedo"] = sample["diffuse_color"] + sample["glossy_color"]
        
        if "depth" in self.low_spp_aux_features:
            sample["depth"] = sample["depth"][:3, ...]

        self.all_low_spp.append(sample)

    def load_high_spp_sample(self, high_batch_path, filename):
        sample = {}
        rgb_path = os.path.join(high_batch_path, filename)
        rgb_data = self.image_to_tensor(Image.open(rgb_path).convert("RGB"))
        sample["rgb"] = rgb_data

        self.all_high_spp.append(sample)

    def preprocess_samples(self):
        pass

    def __len__(self):
        return len(self.all_low_spp)

    def __getitem__(self, idx):
        low_spp_features = self.all_low_spp[idx]
        high_spp_features = self.all_high_spp[idx]

        return low_spp_features, high_spp_features


def show_images(dataloader, num_images=6):
    # Get a batch of training data
    data_iter = iter(dataloader)
    features, targets = next(data_iter)

    print(features["rgb"])

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
