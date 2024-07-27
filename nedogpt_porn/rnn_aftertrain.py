import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
        # Проверяем размерности
        #print("RNN output shape:", out.shape)
        out = self.fc(out[:, -1, :])  # Используем последний временной шаг
        return out

vocab_size = 1
asel = torch.load("RNN_degenerat.pth")
model = CharRNN(vocab_size=asel["vocab_size"], embed_size=64, hidden_size=128, num_layers=2)
model.eval()
#model = CharRNN(vocab_size, embed_size=64, hidden_size=128, num_layers=2)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def generate_text(model, start_str, length=100):
    model.eval()
    chars = list(start_str)
    input_seq = torch.tensor([asel['char_to_idx'][c] for c in start_str], dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq)
            _, predicted_idx = torch.max(output, dim=1)
            predicted_char = asel['idx_to_char'][predicted_idx.item()]
            chars.append(predicted_char)
            input_seq = torch.cat([input_seq[:, 1:], predicted_idx.unsqueeze(0)], dim=1)
    
    return ''.join(chars)

context = ""
out = ""
while 1:
    inp = input("Напиши чота:")
    context += inp + "\n"
    out += generate_text(model, context, 100) + "\n"
    print(out)
    context += out + "\n"
