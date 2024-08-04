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
        transforms.Resize((180, 320)),
        transforms.ToTensor()
    ])

    train_dataset = PairedImageDataset(
        degraded_dir='train/degraded', 
        clean_dir='train/clean', 
        transform=transform, 
        augment=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)

    val_dataset = PairedImageDataset(
        degraded_dir='val/degraded', 
        clean_dir='val/clean', 
        transform=transform
    )
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    model = VolterraNet(num_channels=3)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    patience = 10
    best_loss = float('inf')
    epochs_without_improvement = 0

    scaler = GradScaler()
    accumulation_steps = 4

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for degraded_images, clean_images in train_dataloader:
            degraded_images, clean_images = degraded_images.to(device), clean_images.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                restored_images = model(degraded_images)
                loss = criterion(restored_images, clean_images)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            running_loss += loss.item()
        
        average_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_degraded_images, val_clean_images in val_dataloader:
                val_degraded_images, val_clean_images = val_degraded_images.to(device), val_clean_images.to(device)
                with autocast(device_type='cuda'):
                    val_restored_images = model(val_degraded_images)
                    val_loss += criterion(val_restored_images, val_clean_images).item()
        
        average_val_loss = val_loss / len(val_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}')

        scheduler.step(average_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_volterra_net.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                torch.save(model.state_dict(), 'best_volterra_net.pth')
                print("Early stopping triggered")
                break
