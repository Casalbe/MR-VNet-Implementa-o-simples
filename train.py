import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torchvision.transforms import functional as TF
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

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
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        self.conv2_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_channels)
        self.bn2_2 = nn.BatchNorm2d(num_channels)
        
        self.conv3_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(num_channels)
        self.bn3_2 = nn.BatchNorm2d(num_channels)
        self.bn3_3 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        linear_term = self.bn1(self.conv1(x))
        
        quad_term = torch.mul(self.bn2_1(self.conv2_1(x)), self.bn2_2(self.conv2_2(x))).clamp(min=0)
        
        cubic_term = torch.mul(self.bn3_1(self.conv3_1(x)), 
                               torch.mul(self.bn3_2(self.conv3_2(x)), self.bn3_3(self.conv3_3(x)))).clamp(min=0)
        
        out = linear_term + quad_term + cubic_term
        return out

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.lpips = lpips.LPIPS(net='vgg').to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, restored_image, clean_image):
        return self.lpips(restored_image, clean_image)

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
    vgg_loss = VGGPerceptualLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    
    lpips_metric = lpips.LPIPS(net='vgg').to(device)

    scaler = GradScaler()
    accumulation_steps = 4

    num_epochs = 50
    best_loss = float('inf')
    patience = 10
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (degraded_images, clean_images) in enumerate(train_dataloader):
            degraded_images, clean_images = degraded_images.to(device), clean_images.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                restored_images = model(degraded_images)
                mse_loss = criterion(restored_images, clean_images)
                perceptual_loss = vgg_loss(restored_images, clean_images)
                
                loss = (mse_loss + perceptual_loss).mean() / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item()
        
        average_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}')

        model.eval()
        val_loss = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0
        with torch.no_grad():
            for val_degraded_images, val_clean_images in val_dataloader:
                val_degraded_images, val_clean_images = val_degraded_images.to(device), val_clean_images.to(device)
                with autocast(device_type='cuda'):
                    val_restored_images = model(val_degraded_images)
                    val_loss += criterion(val_restored_images, val_clean_images).item()
                    psnr_total += psnr_metric(val_restored_images, val_clean_images)
                    ssim_total += ssim_metric(val_restored_images, val_clean_images)
                    lpips_total += lpips_metric(val_restored_images, val_clean_images).mean()

        average_val_loss = val_loss / len(val_dataloader)
        average_psnr = psnr_total / len(val_dataloader)
        average_ssim = ssim_total / len(val_dataloader)
        average_lpips = lpips_total / len(val_dataloader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], PSNR: {average_psnr:.4f} dB, SSIM: {average_ssim:.4f}, LPIPS: {average_lpips:.4f}')

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'best_volterra_net.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break
