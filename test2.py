import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import NoiseDataset
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import NoiseDataset
from model import UNet
import numpy as np
from processing import display_mask


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def visualize_image_and_noise(dataset, index=0):
    img, noise = dataset[index]
    img = img.numpy().transpose((1, 2, 0))  # Convert from CxHxW to HxWxC
    noise = noise.numpy().transpose((1, 2, 0))
    
    # Normalize the image for displaying
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(noise)
    plt.title('Noise')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    # Load configuration
    config_path = 'config.yaml'
    config = load_config(config_path)
    
    # Dataset Preparation using the config

    dataset = NoiseDataset(images_dir=config['dataset']['images_dir'], masks_dir=config['dataset']['masks_dir'])
    img, noise = dataset[0]
    display_mask(img)  # You can change the index to see different images
    display_mask(noise)
