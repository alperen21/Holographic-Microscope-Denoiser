from model import UNet
from dataset import NoiseDataset
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_ssim


criterion_mse = nn.MSELoss()
criterion_ssim = pytorch_ssim.SSIM(window_size=11)

def combined_loss(y_true, y_pred):
    mse_loss = criterion_mse(y_true, y_pred)
    ssim_loss = 1 - criterion_ssim(y_true, y_pred)
    return mse_loss + ssim_loss

def main():
    images_dir = './images/'
    masks_dir = './masks/'
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    dataset = NoiseDataset(images_dir, masks_dir, transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
    num_epochs = 50
    best_loss = float('inf')
    
    model = UNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)



    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for noisy_img, noise in train_loader:
            noisy_img = noisy_img.to(device)
            noise = noise.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_img)
            loss = combined_loss(noise, outputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * noisy_img.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy_img, noise in val_loader:
                noisy_img = noisy_img.to(device)
                clean_img = noise.to(device)
                outputs = model(noisy_img)
                loss = combined_loss(noise, outputs)
                val_loss += loss.item() * noisy_img.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for noisy_img, noise in test_loader:
            noisy_img = noisy_img.to(device)
            clean_img = noise.to(device)
            outputs = model(noisy_img)
            loss = combined_loss(noise, outputs)
            test_loss += loss.item() * noisy_img.size(0)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')


if __name__ == "__main__":
    main()