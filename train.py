import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import NoiseDataset
from model import UNet

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and config['device']['use_cuda'] else "cpu")

    # Dataset Preparation using the config
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config['dataset']['transform']['resize']),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['dataset']['transform']['mean'], std=config['dataset']['transform']['std'])
    ])

    dataset = NoiseDataset(images_dir=config['dataset']['images_dir'], masks_dir=config['dataset']['masks_dir'], transform=transform)

    # Splitting the dataset into training and test sets
    train_size = int(config['training']['train_test_split_ratio'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Model Initialization
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])

    # Training and Testing
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        for imgs, noises in train_loader:
            imgs, noises = imgs.to(device), noises.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, noises)  # Assuming you want to reconstruct the original images
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        # Testing loop
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for imgs, noises in test_loader:
                imgs, noises = imgs.to(device), noises.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, noises)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), config['paths']['model_save_path'])

if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)
    main(config)
