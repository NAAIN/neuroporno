import torch
import torch.nn as nn
import numpy as np
import os

model_file = "models/GRU_degenerat_12000_1e-05.pth"
length=100
temperature=2
top_k=100

class CharGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CharGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
def generate_text(model, start_str, char_to_idx, idx_to_char, length, temperature, top_k):
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
            if predicted_idx in idx_to_char:
                predicted_char = idx_to_char[predicted_idx]
            else:
                predicted_char = ' '

            chars.append(predicted_char)
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted_idx]], dtype=torch.long)], dim=1)
    
    return ''.join(chars)
if os.path.isfile(model_file):
    checkpoint = torch.load(model_file)
    vocab_size = checkpoint['vocab_size']
    embed_size = checkpoint['embed_size']
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    model = CharGRU(vocab_size, embed_size, hidden_size, num_layers)
    saved_vocab_size = checkpoint['vocab_size']
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
        model.load_state_dict(checkpoint['model_state_dict'])
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    model.eval()
else:
    print("а модель")
    exit()
try:
    while 1:
        start_str = input("напиши чонить ")
        if len(start_str) < 1:
            start_str = " "
        try:
            generated_text = generate_text(model, start_str, char_to_idx=checkpoint['char_to_idx'], idx_to_char=checkpoint['idx_to_char'], length=100, temperature=0.8, top_k=5)
            print(generated_text)
        except KeyError as e:
            print(f"молодой чоловек тут как бы {e}")
except KeyboardInterrupt:
    exit("ну ладно")
