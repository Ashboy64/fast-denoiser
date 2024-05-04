import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision.transforms import functional as F
from noise import AddGaussianNoise


class ImageDataset(Dataset):
    def __init__(self, folder_path, noise_level=0.1):
        # Store the file names of the images
        self.file_names = [os.path.join(folder_path, f) for f in
                           os.listdir(folder_path) if f.endswith('.jpg')]
        self.noise_level = noise_level
        # Define the transform for the input images (resize only)
        self.input_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            AddGaussianNoise(mean=0, std=noise_level),
            
        ])
        # Define the transform for the target images (resize and color jitter)
        self.target_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # Load an image from the file
        image = Image.open(self.file_names[index])
        
        # Apply the transformations
        input_image = self.input_transform(image)
        target_image = self.target_transform(image)
        
        return input_image, target_image


def show_images(dataloader, num_images=6):
    # Get a batch of training data
    data_iter = iter(dataloader)
    images, targets = next(data_iter)

    # Convert tensors to grid of images
    input_grid = vutils.make_grid(images[:num_images], nrow=3, normalize=True, scale_each=True)
    target_grid = vutils.make_grid(targets[:num_images], nrow=3, normalize=True, scale_each=True)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    
    axs[0].imshow(input_grid.permute(1, 2, 0))  # permute to convert tensor format from CxHxW to HxWxC
    axs[0].set_title('Input Images')
    axs[0].axis('off')
    
    axs[1].imshow(target_grid.permute(1, 2, 0))
    axs[1].set_title('Ground Truth Images')
    axs[1].axis('off')
    
    plt.show()


if __name__ == '__main__':
    # Usage
    folder_path = 'baseline-dataset/Images'
    dataset = ImageDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # To see the effects, you can loop through the DataLoader
    for inputs, targets in dataloader:
        # should print shapes like [batch_size, 3, 64, 64]
        print(inputs.shape, targets.shape)
        break

    show_images(dataloader, num_images=6)