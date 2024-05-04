from model import DenoisingCNN
from dataset import ImageDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

IMAGES_PATH = "baseline-dataset/TrainImages/"

def train(model, dataloader, loss_fn, optimizer, num_epochs):
    model.train()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, (noisy_imgs, clean_imgs) in enumerate(dataloader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # Forward pass: compute the predicted outputs
            outputs = model(noisy_imgs)
            loss = loss_fn(outputs, clean_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 10 == 0:
                print("Finished", i+1, "examples.")
        
        # Print average loss for this epoch
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')
        torch.save(model, f"checkpoints/epoch{epoch+1}.pth")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    model = DenoisingCNN().to(device)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0)
    model.apply(init_weights)
    dataset = ImageDataset(IMAGES_PATH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    train(model, dataloader, loss_fn, optimizer, num_epochs)



if __name__ == '__main__':
    main()