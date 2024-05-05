from omegaconf import OmegaConf

import torch

from datasets import load_dataset
from data.utils import *


def load_from_disk(
    folder_path, noise_level, batch_size, train_frac, val_frac, test_frac
):
    raw_dataset = SavedImagesDataset(folder_path)
    split_datasets = torch.utils.data.random_split(
        raw_dataset, lengths=[train_frac, val_frac, test_frac]
    )

    train_loader = DataLoader(
        NoiseWrapper(split_datasets["train"], noise_level),
        batch_size=batch_size,
        shuffle=True,
    )
    val_train = DataLoader(
        NoiseWrapper(split_datasets["val"], noise_level),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        NoiseWrapper(split_datasets["test"], noise_level),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_train, test_loader


def load_tiny_imagenet(
    test_frac=0.1,
    seed=0,
    batch_size=64,
    noise_level=0.1,
    num_preprocessing_workers=8,
    num_dataloader_workers=8,
    **kwargs
):
    # Load from Huggingface and split train set into train and test.
    raw_tiny_imagenet = load_dataset("Maysee/tiny-imagenet")
    tiny_imagenet = raw_tiny_imagenet["train"].train_test_split(
        test_size=test_frac, seed=seed, shuffle=True
    )
    tiny_imagenet["val"] = raw_tiny_imagenet["valid"]

    # Only keep examples with three channels.
    tiny_imagenet = tiny_imagenet.filter(
        lambda example: example["image"].mode == "RGB"
    )

    # Rename the "image" feature to "rgb", and "label" to "class".
    def rename_features(example):
        return {"rgb": example["image"], "class": example["label"]}

    tiny_imagenet = tiny_imagenet.map(
        rename_features,
        remove_columns=["image", "label"],
        desc="Renaming features",
        num_proc=num_preprocessing_workers,
    )

    # Wrap with NoiseWrapper and return dataloaders.
    train_loader = DataLoader(
        NoiseWrapper(tiny_imagenet["train"], noise_level),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_dataloader_workers,
    )
    val_train = DataLoader(
        NoiseWrapper(tiny_imagenet["val"], noise_level),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
    )
    test_loader = DataLoader(
        NoiseWrapper(tiny_imagenet["test"], noise_level),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
    )

    return train_loader, val_train, test_loader


def load_data(data_config):
    if data_config.name == "tiny_imagenet":
        return load_tiny_imagenet(**data_config)
    elif "saved" in data_config.name:
        return load_from_disk(**data_config)
    return None


if __name__ == "__main__":
    data_config = OmegaConf.create(
        {
            "name": "tiny_imagenet",
            "num_proc": 8,
            "seed": 0,
            "batch_size": 64,
        }
    )

    dataloaders = load_data(data_config)
    show_images(dataloaders["train"], num_images=6)
