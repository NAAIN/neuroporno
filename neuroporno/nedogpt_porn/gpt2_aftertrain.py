import random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузите модель и токенизатор
model = GPT2LMHeadModel.from_pretrained('resources/gopota_files/models/gopota')
tokenizer = GPT2Tokenizer.from_pretrained('resources/gopota_files/models/gopota')

import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Установите seed
set_seed(42)

# Функция для генерации текста
def generate_text(prompt, max_length=50, seed=42):
    # Установите seed внутри функции генерации
    set_seed(seed)
    
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)

    # Генерация текста
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,  # Использовать sample для случайной генерации
            top_k=5000,        # Оставить только топ-K токенов для выбора
            top_p=0.95       # Оставить только самые вероятные токены до достижения вероятности p
        )
    
    # Декодирование и вывод
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while 1:
    set_seed(int(random.random()))
    print(generate_text(input("напиши чонить ")))