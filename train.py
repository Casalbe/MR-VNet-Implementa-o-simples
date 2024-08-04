import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from torchvision.transforms import functional as TF

class PairedImageDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, transform=None, augment=False):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.augment = augment
        self.degraded_images = [f for f in os.listdir(degraded_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.clean_images = [f for f in os.listdir(clean_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.degraded_images.sort()
        self.clean_images.sort()
    
    def __len__(self):
        return len(self.degraded_images)
    
    def __getitem__(self, idx):
        degraded_img_name = os.path.join(self.degraded_dir, self.degraded_images[idx])
        clean_img_name = os.path.join(self.clean_dir, self.clean_images[idx])
        degraded_image = Image.open(degraded_img_name).convert('RGB')
        clean_image = Image.open(clean_img_name).convert('RGB')

        if self.augment:
            if torch.rand(1) > 0.5:
                degraded_image = TF.hflip(degraded_image)
                clean_image = TF.hflip(clean_image)
            if torch.rand(1) > 0.5:
                degraded_image = TF.vflip(degraded_image)
                clean_image = TF.vflip(clean_image)
        
        if self.transform:
            degraded_image = self.transform(degraded_image)
            clean_image = self.transform(clean_image)
            
        return degraded_image, clean_image

class VolterraNet(nn.Module):
    def __init__(self, num_channels):
        super(VolterraNet, self).__init__()
        self.num_channels = num_channels
        
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        linear_term = self.conv1(x)
        quad_term = self.conv2_1(x) * self.conv2_2(x)
        cubic_term = self.conv3_1(x) * self.conv3_2(x) * self.conv3_3(x)
        out = linear_term + quad_term + cubic_term
        return out

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    transform = transforms.Compose([
        transforms.Resize((320, 180)),
        transforms.ToTensor()
    ])

    dataset = PairedImageDataset(
        degraded_dir='train/degraded', 
        clean_dir='train/clean', 
        transform=transform, 
        augment=True
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)


    model = VolterraNet(num_channels=3)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    patience = 10
    best_loss = float('inf')
    epochs_without_improvement = 0

    scaler = GradScaler()

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for degraded_images, clean_images in dataloader:
            degraded_images, clean_images = degraded_images.to(device), clean_images.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                restored_images = model(degraded_images)
                loss = criterion(restored_images, clean_images)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        average_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

        torch.save(model.state_dict(), 'best_volterra_net.pth')
