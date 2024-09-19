import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((180,320)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)

def save_image(tensor, path):
    image = tensor.squeeze(0).cpu().clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)

model = VolterraNet(num_channels=3)
model.load_state_dict(torch.load('backup/best_volterra_net.pth'))

model.eval()

degraded_image_path = 'image.jpg'
restored_image_path = 'deblured.jpg'

degraded_image = load_image(degraded_image_path)

with torch.no_grad():  
    restored_image = model(degraded_image)

save_image(restored_image, restored_image_path)

original_image = Image.open(degraded_image_path)
restored_image_disp = Image.open(restored_image_path)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Imagem degradada")
plt.imshow(original_image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagem restaurada")
plt.imshow(restored_image_disp)
plt.axis("off")

plt.show()
