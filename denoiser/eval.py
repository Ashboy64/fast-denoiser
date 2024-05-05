import os
import tqdm
import hydra
import matplotlib.pyplot as plt

import random
import numpy as np

import torch

from data import load_data, move_features_to_device
from models import load_model


def show_image(denoised_image, noisy_image, original_image):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Noisy Image", "Denoised Output", "Ground Truth"]
    images = [noisy_image, denoised_image, original_image]

    for idx, (img, title) in enumerate(zip(images, titles)):
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ax[idx].imshow(img)
        ax[idx].set_title(title)
        ax[idx].axis("off")
    plt.show()


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def visualize_predictions(model, dataloader, device, num_images=10):
    features, targets = next(iter(dataloader))
    features = move_features_to_device(features, device)
    targets = targets.to(device)

    outputs = model(features)

    for image_idx in range(num_images):
        original_image = targets[image_idx, ...]
        noisy_image = features["rgb"][image_idx, ...]
        denoised_image = outputs[image_idx, ...]

        show_image(denoised_image, noisy_image, original_image)


@hydra.main(config_path="config", config_name="baseline", version_base=None)
@torch.inference_mode()
def main(config):
    seed(config.seed)

    _, val_loader, test_loader = load_data(config.data)

    model = load_model(config.model).to(config.device)
    model.load_state_dict(torch.load(config.logging.ckpt_dir))
    model.eval()

    visualize_predictions(model, val_loader, config.device)


if __name__ == "__main__":
    main()
