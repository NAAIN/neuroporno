import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
transform = transforms.Compose([
    #transforms.Resize((1024, 1024)),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
dataset = CustomDataset(img_dir='/mnt/ata-ST500LM030/neironke/ComfyUI-master/output', transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        # self.hidden = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        # )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = self.hidden(x)
        x = self.decoder(x)
        return x


model = Autoencoder()
if os.path.isfile('img2img_degenerat.pth'): model = torch.load('img2img_degenerat.pth')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30

try:
    for epoch in range(num_epochs):
        for images in dataloader:
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}', end='\r')
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
except KeyboardInterrupt:
    print("та за шо((")
torch.save(model,"img2img_degenerat.pth")
