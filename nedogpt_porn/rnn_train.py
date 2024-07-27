import torch
from torch.utils.data import Dataset, DataLoader
import os.path

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

dataset = TextDataset('dset.txt')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

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
        out = self.fc(out[:, -1, :])  # Используем последний временной шаг
        return out

embed_size = 32
hidden_size = 32
num_layers = 10
batch_size = 420
learn_rate = 0.1
num_epochs = 10000

if os.path.isfile("RNN_degenerat.pth"):
    checkpoint = torch.load("RNN_degenerat.pth")
    saved_vocab_size = checkpoint['vocab_size']
    current_vocab_size = len(dataset.chars)
    if current_vocab_size != saved_vocab_size:
        print(f"Размер словаря изменился с {saved_vocab_size} на {current_vocab_size}.")
        # Создаем новую модель с текущим размером словаря
        model = CharRNN(vocab_size=current_vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        # Загружаем параметры, которые совпадают по размеру
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
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}', end='\r')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
except KeyboardInterrupt:
    print("не трогайте сука9(9((")
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'char_to_idx': dataset.char_to_idx,
    #     'idx_to_char': dataset.idx_to_char,
    #     'vocab_size': len(dataset.chars)
    # }, "RNN_degenerat.pth")
    
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

torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': dataset.char_to_idx,
    'idx_to_char': dataset.idx_to_char,
    'vocab_size': len(dataset.chars)
}, "RNN_degenerat.pth")
print(generate_text(model, 'Привет, ЛУКОЙЛ??', 100))
