import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

# Загрузите модель и токенизатор
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Установите токен паддинга
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Загрузите датасет
dataset = load_dataset('text', data_files={'train': 'resources/gopota_files/gopota_dataset/dset.txt'})

# Токенизируем данные
def tokenize_function(examples):
    encodings = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    encodings['labels'] = encodings['input_ids']
    return encodings

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Создайте класс Dataset для DataLoader
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = torch.tensor(encodings['input_ids'])
        self.attention_mask = torch.tensor(encodings['attention_mask'])
        self.labels = torch.tensor(encodings['labels'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Параметры обучения
batch_size = 3
num_epochs = 1
learning_rate = 0.001

# Создайте DataLoader
train_dataset = TextDataset(tokenized_datasets['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Оптимизатор
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Перенос модели на GPU (если доступен)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Обучение
model.train()
try:
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Average Loss for Epoch {epoch + 1}: {avg_loss}')
except KeyboardInterrupt:
    print("не трогайте сука9(9((")
    
# Сохранение модели
model.save_pretrained('resources/gopota_files/models/gopota')
tokenizer.save_pretrained('resources/gopota_files/models/gopota')
