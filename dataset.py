import os
import numpy as np
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from processing import get_combined_mask, display_mask, overlay_mask_on_image


class NoiseDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        self.mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.txt')]
        self.image_files.sort()
        self.mask_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = get_combined_mask(mask_path, image.shape)
        
        masked_image = overlay_mask_on_image(image, mask)
        noise = cv2.subtract(image, masked_image)

        if self.transform:
            image = self.transform(image)
            noise = self.transform(noise)
        
        
         # Convert to PIL Image


        return image, noise
