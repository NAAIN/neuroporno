import torch
from torch.utils.data import Dataset, DataLoader
import os.path
# КРУТИЛАЧКИИИ
embed_size = 128
hidden_size = 128
num_layers = 12
#^^^^^^^^^^^^^^^^настройки НОВОЙ обучаемой модели(если доучивать то эти крутилки не учитываются)
batch_size = 100000
learn_rate = 0.00001
num_epochs = 12000
datasetFile = "resources/gopota_files/dset.txt"
ExistsModelPthName = ""
#ExistsModelPthName = "models/RNN/RNN_degenerat_начал_чота_понимать.pth" #закомментируй чтобы создать новую модель
LearnedPthName = "neuroporno/models/RNN/RNN_degenerat.pth"
start_str = "жопа"
debug_in_name = True #Добавляет колво эпох, learn rate и параметры слоёв в название файла
early_stop = False
shuffle = True
early_stopping_patience = 10
early_stopping_counter = 0
early_stopping_factor = 0.1
#============================
#Код ниже лучше не трогать (говнокод warning)
class TextDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.text = f.read()
        self.chars = list(set(self.text))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.data = [self.char_to_idx[char] for char in self.text]

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.data[idx + 1], dtype=torch.long)

dataset = TextDataset(datasetFile)
dataloader = DataLoader(dataset, batch_size,shuffle)

import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CharRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

if os.path.isfile(ExistsModelPthName):
    checkpoint = torch.load(ExistsModelPthName)
    saved_vocab_size = checkpoint['vocab_size']
    current_vocab_size = len(dataset.chars)
    if current_vocab_size != saved_vocab_size:
        print(f"Размер словаря изменился с {saved_vocab_size} на {current_vocab_size}.")
        model = CharRNN(vocab_size=current_vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        
        for name, param in state_dict.items():
            if name in model_state_dict:
                if param.size() == model_state_dict[name].size():
                    model_state_dict[name].copy_(param)
                else:
                    print(f"Пропуск параметра {name} из-за несоответствия размеров: {param.size()} vs {model_state_dict[name].size()}")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        model = CharRNN(vocab_size=saved_vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
else:
    saved_vocab_size = 1
    model = CharRNN(vocab_size=len(dataset.chars), embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

try:
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}', end='\r')
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')
except KeyboardInterrupt:
    print("не трогайте сука9(9((")
    
def generate_text(model, start_str, length=100):
    model.eval()
    chars = list(start_str)
    input_seq = torch.tensor([dataset.char_to_idx[c] for c in start_str], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq)
            _, predicted_idx = torch.max(output, dim=1)
            predicted_char = dataset.idx_to_char[predicted_idx.item()]
            chars.append(predicted_char)
            input_seq = torch.cat([input_seq[:, 1:], predicted_idx.unsqueeze(0)], dim=1)
    
    return ''.join(chars)

if debug_in_name: 
    import time
    LearnedPthName += f"{epoch}_{learn_rate}_{embed_size}_{hidden_size}_{num_layers}_{time.time}"

torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': dataset.char_to_idx,
    'idx_to_char': dataset.idx_to_char,
    'vocab_size': len(dataset.chars),
    "embed_size": embed_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
}, LearnedPthName)

print(generate_text(model, 'Привет, ЛУКОЙЛ??', 100))
