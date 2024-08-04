import torch
import torch.nn as nn

model_file = "neuroporno/models/LSTM/LSTM_degenerat.pth6_0.0001_128_128_12_1722751339.2929528"
length=100

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

vocab_size = 1
asel = torch.load(model_file)
model = CharRNN(vocab_size=asel["vocab_size"], embed_size=asel["emded_size"], hidden_size=asel["hidden_size"], num_layers=asel["num_layers"])
model.eval()

def generate_text(model, start_str, length):
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


try:
    while 1:
        start_str = input("напиши чонить ")
        if len(start_str) < 1:
            start_str = " "
        try:
            generated_text = generate_text(model, start_str, length)
            print(generated_text)
        except KeyError as e:
            print(f"молодой чоловек тут как бы KeyError")
except KeyboardInterrupt:
    exit("ну ладно")