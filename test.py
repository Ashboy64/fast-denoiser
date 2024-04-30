import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from noise import AddGaussianNoise
from model import DenoisingCNN
import torch.nn as nn

TEST_FOLDER = "baseline-dataset/TestImages/"

def preprocess(image_file: str):
    noise_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0, std=0.1)
    ])
    original_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    image = Image.open(image_file)
    original_image = original_transform(image)
    original_image.unsqueeze(0)
    noisy_image = noise_transform(image)
    noisy_image = noisy_image.unsqueeze(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return noisy_image.to(device), original_image.to(device)


def denoise_image(model, image_file: str):
    noisy_image, original_image = preprocess(image_file)
    model.eval()
    with torch.no_grad():
        denoised_image = model(noisy_image)
    return denoised_image, noisy_image, original_image


def show_image(denoised_image, noisy_image, original_image):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Noisy Image', 'Denoised Output', 'Ground Truth']
    images = [noisy_image, denoised_image, original_image]

    for idx, (img, title) in enumerate(zip(images, titles)):
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ax[idx].imshow(img)
        ax[idx].set_title(title)
        ax[idx].axis('off')
    plt.show()


def main():
    image_file = "baseline-dataset/TestImages/3637013_c675de7705.jpg"
    model = torch.load("checkpoints/epoch9.pth")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    model.eval()
    denoised_image, noisy_image, original_image = denoise_image(model, image_file)
    show_image(denoised_image, noisy_image, original_image)


if __name__ == "__main__":
    main()

