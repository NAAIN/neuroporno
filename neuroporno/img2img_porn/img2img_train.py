import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
# КРУТИЛАЧКИИИ
batch_size = 1 #лучше по одной, хоть модель поумнее будет (наверн)
learn_rate = 0.001
num_epochs = 1200
datasetDir = "resources/img2img_dataset/"
ExistsModelPthName = ""
#ExistsModelPthName = "models/img2img_degenerat.pth" #закомментируй чтобы создать новую модель
LearnedPthName = "models/img2img_degenerat.pth"
debug_in_name = True
Shuffle = True
#^^^^^^^^^^^^^^^^^^^^^^^^^^ 
# Код ниже лучше не трогать (говнокод warning)
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
    transforms.Resize((1024, 1024)),
    #transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
dataset = CustomDataset(datasetDir,transform)
dataloader = DataLoader(dataset, batch_size,Shuffle)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
if os.path.isfile(ExistsModelPthName): model = torch.load(ExistsModelPthName)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),learn_rate)

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
if debug_in_name: LearnedPthName += f"{epoch}_{learn_rate}"
torch.save(model,"models/img2img_degenerat.pth")
