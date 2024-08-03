import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
# КРУТИЛАЧКИИИ
embed_size = 128
hidden_size = 128
num_layers = 12
#^^^^^^^^^^^^^^^^настройки НОВОЙ обучаемой модели(если доучивать то эти крутилки не учитываются)
batch_size = 100
learn_rate = 0.00001
num_epochs = 12000
datasetFile = "resources/gopota_files/dset.txt"
ExistsModelPthName = ""
#ExistsModelPthName = "models/GRU_degenerat_начал_чота_понимать.pth" #закомментируй чтобы создать новую модель
LearnedPthName = "neuroporno/models/GRU/GRU_degenerat.pth"
start_str = "жопа"
debug_in_name = True #Добавляет колво эпох, learn rate и параметры слоёв в название файла
early_stop = False
early_stopping_patience = 10
early_stopping_counter = 0
early_stopping_factor = 0.1
#============================
#Код ниже лучше не трогать (говнокод warning)

class CharGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(CharGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -1, 1)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

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

def generate_text(model, start_str, char_to_idx, idx_to_char, length=100, temperature=1.0, top_k=5):
    model.eval()
    chars = list(start_str)
    input_seq = torch.tensor([char_to_idx[c] for c in start_str], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq)
            output = output / temperature
            probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            
            top_k_indices = np.argsort(probabilities)[-top_k:]
            top_k_probs = probabilities[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)
            predicted_idx = np.random.choice(top_k_indices, p=top_k_probs)
            predicted_char = idx_to_char[predicted_idx]

            chars.append(predicted_char)
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted_idx]], dtype=torch.long)], dim=1)
    
    return ''.join(chars)

dataset = TextDataset(datasetFile)
dataloader = DataLoader(dataset, batch_size)
vocab_size = len(dataset.chars)

if os.path.isfile(ExistsModelPthName):
    checkpoint = torch.load(ExistsModelPthName)
    saved_vocab_size = checkpoint['vocab_size']
    embed_size=checkpoint['embed_size']
    hidden_size=checkpoint['hidden_size']
    num_layers=checkpoint['num_layers']
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    if saved_vocab_size != vocab_size:
        print(f"Размер словаря изменился с {saved_vocab_size} на {vocab_size}.")
        model = CharGRU(vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
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
        model = CharGRU(vocab_size, embed_size, hidden_size, num_layers)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
else:
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char
    model = CharGRU(vocab_size, embed_size, hidden_size, num_layers)

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learn_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=early_stopping_factor, patience=early_stopping_patience)
min_loss = float('inf')

try:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss}',end='\r')
        epoch_loss /= len(dataloader)
        scheduler.step(epoch_loss)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss}')

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience and early_stop:
            print("Early stopping")
            break
except KeyboardInterrupt:
    print("\nне трогайте сука9(9((")
    
if debug_in_name: 
    import time
    LearnedPthName += f"{epoch}_{learn_rate}_{embed_size}_{hidden_size}_{num_layers}_{time.time()}"
    
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size,
    "embed_size": embed_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers,
}, LearnedPthName)

generated_text = generate_text(model, start_str, char_to_idx, idx_to_char, length=100, temperature=0.7, top_k=10)
print(generated_text)
print(f"Model saved to {LearnedPthName}")
