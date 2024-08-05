from scipy.io import wavfile
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random

X = torch.tensor([1,2,1], dtype=torch.float32)
iterateR = 0.0
ASample = 0.0
output = []

start_time = time.time()

orig_rate,orig = wavfile.read("resources/wav_porn_files/orig.wav")
if orig.ndim == 2:
    print("Warning, Original file is Stereo. converting to mono..")
    audio_data = np.mean(orig, axis=1).astype(orig.dtype)
print("Orig len:  ",len(orig))

audio_data = np.frombuffer(audio_data, dtype=np.int16)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.a = nn.ReLU()
        self.linear = nn.Linear(2, 10)
        self.linear1 = nn.Linear(10, 1)
        self.linear11 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.a(x)
        x = self.linear1(x)
        x = self.a(x)
        x = self.linear11(x)
        return self.linear2(x)

model = SimpleNet()
print(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

number_of_epochs = 300000
for epoch in range(number_of_epochs):
    iterateR = epoch / 10000000
    ASample = audio_data[epoch] / 100000
    X = torch.tensor([float(ASample),float(random.random() % 1)], dtype=torch.float32)
    pred_x = model(X)
    output.append(pred_x.item())
    loss = criterion(pred_x, X)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
       print(f'Epoch {epoch}, Loss: {loss.item()} X:{X}, pred_x:{pred_x}')

print("====Make prediction ====")
test_x = torch.tensor([float(audio_data[epoch]),iterateR], dtype=torch.float32)
y = model(test_x)
print(test_x)
print(y)
torch.save(model, "resources/wav_porn_files/models/wav_degenerat.pth")