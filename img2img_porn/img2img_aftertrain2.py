import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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
model = torch.load('models/img2img_degenerat.pth')

transform = transforms.Compose([
    #transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

image = Image.open('resources/a.png').convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)

output_image = output.squeeze(0)
output_image = transforms.ToPILImage()(output_image)
output_image.save('resources/a_out.png')