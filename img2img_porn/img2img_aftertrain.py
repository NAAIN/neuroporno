import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

inputImage = "resources/a.png"
outputImage = "resources/a_out.png"
modelFile = "models/img2img_degenerat.pth"
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
model = torch.load(modelFile)
#model.eval()

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# Загрузка и преобразование изображения
image = Image.open(inputImage).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad(): output = model(image)

# Преобразование и сохранение выходного изображения
output = output.squeeze(0)
output = transforms.ToPILImage()(output)
output.save(outputImage)