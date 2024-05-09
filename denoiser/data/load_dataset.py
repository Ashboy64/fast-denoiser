import sys

sys.path.append("..")

from omegaconf import OmegaConf

import torch

from datasets import load_dataset
from data.utils import *


DATASETS = {}


def register_dataset(name):
    def register_curr_dataset(dataset_fn):
        DATASETS[name] = dataset_fn
        return dataset_fn

    return register_curr_dataset


@register_dataset("pbrt")
def load_pbrt_data(
    folder_path,
    batch_size,
    train_frac,
    val_frac,
    test_frac,
    num_dataloader_workers=8,
    **kwargs
):
    
    print(f"USING BATCH SIZE {batch_size}")

    raw_dataset = PBRT_Dataset(folder_path)
    split_datasets = torch.utils.data.random_split(
        raw_dataset, lengths=[train_frac, val_frac, test_frac]
    )

    train_loader = DataLoader(
        split_datasets[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_dataloader_workers,
    )
    val_train = DataLoader(
        split_datasets[1],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
    )
    test_loader = DataLoader(
        split_datasets[2],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
    )

    return train_loader, val_train, test_loader


@register_dataset("disk_gaussian_noise")
def load_from_disk(
    folder_path,
    noise_level,
    batch_size,
    train_frac,
    val_frac,
    test_frac,
    num_dataloader_workers=8,
    **kwargs
):
    raw_dataset = SavedImagesDataset(folder_path)
    split_datasets = torch.utils.data.random_split(
        raw_dataset, lengths=[train_frac, val_frac, test_frac]
    )

    train_loader = DataLoader(
        NoiseWrapper(split_datasets[0], noise_level),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_dataloader_workers,
    )
    val_train = DataLoader(
        NoiseWrapper(split_datasets[1], noise_level),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
    )
    test_loader = DataLoader(
        NoiseWrapper(split_datasets[2], noise_level),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
    )

    return train_loader, val_train, test_loader


@register_dataset("tiny_imagenet")
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
    return DATASETS[data_config.name](**data_config)


if __name__ == "__main__":
    # data_config = OmegaConf.create(
    #     {
    #         "name": "tiny_imagenet",
    #         "num_proc": 8,
    #         "seed": 0,
    #         "batch_size": 64,
    #     }
    # )

    data_config = OmegaConf.create(
        {
            "name": "pbrt",
            "folder_path": "../../rendered_images",
            "train_frac": 0.8,
            "val_frac": 0.1,
            "test_frac": 0.1,
            "num_proc": 8,
            "batch_size": 64,
            "num_dataloader_workers": 2,
        }
    )

    dataloaders = load_data(data_config)

    show_images(dataloaders[0], num_images=8)
