import sys

sys.path.append("..")

from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, ConcatDataset

from datasets import load_dataset
from data.utils import *


DATASETS = {}


def register_dataset(name):
    def register_curr_dataset(dataset_fn):
        DATASETS[name] = dataset_fn
        return dataset_fn

    return register_curr_dataset


def create_dataloaders(
    split_datasets, batch_size, num_dataloader_workers, pin_memory=True
):
    train_loader = DataLoader(
        split_datasets[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_dataloader_workers,
        pin_memory=pin_memory,
        multiprocessing_context="fork",
    )
    val_train = DataLoader(
        split_datasets[1],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
        pin_memory=pin_memory,
        multiprocessing_context="fork",
    )
    test_loader = DataLoader(
        split_datasets[2],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
        pin_memory=pin_memory,
        multiprocessing_context="fork",
    )

    return train_loader, val_train, test_loader


@register_dataset("dummy_pbrt")
def load_dummy_pbrt_data(
    num_examples,
    batch_size,
    train_frac,
    val_frac,
    test_frac,
    num_dataloader_workers=8,
    **kwargs,
):
    raw_dataset = PBRT_DummyDataset(num_examples)
    split_datasets = torch.utils.data.random_split(
        raw_dataset, lengths=[train_frac, val_frac, test_frac]
    )

    return create_dataloaders(
        split_datasets, batch_size, num_dataloader_workers
    )


def load_pbrt_data(
    folder_path,
    low_spp,
    high_spp,
    dtype,
    batch_size,
    preprocess_samples,
    num_dataloader_workers=8,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
    **kwargs,
):
    split_datasets = []

    max_samples = [max_train_samples, max_val_samples, max_test_samples]

    for split_idx, split_name in enumerate(["train", "val", "test"]):
        split_datasets.append(
            PBRT_Dataset(
                folder_path=folder_path,
                split_name=split_name,
                max_samples=max_samples[split_idx],
                low_spp=low_spp,
                high_spp=high_spp,
                dtype=dtype,
                preprocess_samples=preprocess_samples,
            )
        )

    return create_dataloaders(
        split_datasets, batch_size, num_dataloader_workers
    )


@register_dataset("landscape")
def load_landscape_data(**kwargs):
    return load_pbrt_data(**kwargs)


@register_dataset("san_miguel")
def load_san_miguel_data(**kwargs):
    return load_pbrt_data(**kwargs)


def load_blender_data(
    folder_path,
    low_spp,
    high_spp,
    dtype,
    batch_size,
    preprocess_samples,
    num_dataloader_workers=8,
    max_train_samples=None,
    max_val_samples=None,
    max_test_samples=None,
    **kwargs,
):
    split_datasets = []

    max_samples = [max_train_samples, max_val_samples, max_test_samples]

    for split_idx, split_name in enumerate(["train", "val", "test"]):
        split_datasets.append(
            BlenderDataset(
                folder_path=folder_path,
                split_name=split_name,
                max_samples=max_samples[split_idx],
                low_spp=low_spp,
                high_spp=high_spp,
                dtype=dtype,
                preprocess_samples=preprocess_samples,
            )
        )

    return create_dataloaders(
        split_datasets, batch_size, num_dataloader_workers
    )


@register_dataset("classroom")
def load_classroom_data(**kwargs):
    print("LOADING CLASSROOM")
    return load_blender_data(**kwargs)


@register_dataset("bistro")
def load_bistro_data(**kwargs):
    return load_blender_data(**kwargs)


@register_dataset("barbershop")
def load_barbershop_data(**kwargs):
    return load_blender_data(**kwargs)


@register_dataset("hybrid_blender")
def load_hybrid_blender_data(
    config_paths, batch_size, num_dataloader_workers, **kwargs
):
    split_datasets = []

    for split_idx, split_name in enumerate(["train", "val", "test"]):
        split_datasets.append([])

        for config_path in config_paths:
            curr_config = OmegaConf.load(config_path)
            max_samples = [
                curr_config.max_train_samples,
                curr_config.max_val_samples,
                curr_config.max_test_samples,
            ]

            split_datasets[-1].append(
                BlenderDataset(
                    folder_path=curr_config.folder_path,
                    split_name=split_name,
                    max_samples=max_samples[split_idx],
                    low_spp=curr_config.low_spp,
                    high_spp=curr_config.high_spp,
                    dtype=curr_config.dtype,
                    preprocess_samples=curr_config.preprocess_samples,
                )
            )

    split_datasets = [
        ConcatDataset(dataset_list) for dataset_list in split_datasets
    ]

    return create_dataloaders(
        split_datasets, batch_size, num_dataloader_workers
    )


@register_dataset("disk_gaussian_noise")
def load_from_disk(
    folder_path,
    noise_level,
    batch_size,
    train_frac,
    val_frac,
    test_frac,
    num_dataloader_workers=8,
    **kwargs,
):
    raw_dataset = SavedImagesDataset(folder_path)
    split_datasets = torch.utils.data.random_split(
        raw_dataset, lengths=[train_frac, val_frac, test_frac]
    )

    split_datasets = [
        NoiseWrapper(split_dataset, noise_level)
        for split_dataset in split_datasets
    ]

    return create_dataloaders(
        split_datasets, batch_size, num_dataloader_workers
    )


@register_dataset("tiny_imagenet")
def load_tiny_imagenet(
    test_frac=0.1,
    seed=0,
    batch_size=64,
    noise_level=0.1,
    num_preprocessing_workers=8,
    num_dataloader_workers=8,
    **kwargs,
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
        if "label" not in example.keys():
            return {"rgb": example["image"], "class": -1}
        return {"rgb": example["image"], "class": example["label"]}

    tiny_imagenet = tiny_imagenet.map(
        rename_features,
        remove_columns=["image", "label"],
        desc="Renaming features",
        num_proc=num_preprocessing_workers,
    )

    split_datasets = [
        NoiseWrapper(tiny_imagenet[split_name], noise_level)
        for split_name in ["train", "val", "test"]
    ]

    return create_dataloaders(
        split_datasets, batch_size, num_dataloader_workers
    )


def load_data(data_config):
    return DATASETS[data_config.name](**data_config)


if __name__ == "__main__":
    data_config = OmegaConf.create(
        {
            "name": "classroom",
            "folder_path": "../../data/blender/classroom/unzipped",
            "low_spp": 1,
            "high_spp": 1024,
            "dtype": "float32",
            "batch_size": 64,
            "preprocess_samples": True,
            "num_dataloader_workers": 1,
        }
    )

    dataloaders = load_data(data_config)

    show_images(dataloaders[0], num_images=8)
