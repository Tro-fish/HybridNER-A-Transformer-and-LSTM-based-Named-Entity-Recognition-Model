from utils import CONFIG, DotDict
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from NER_dataset import NERDataset
from NER_models import ModelForNER
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

config = DotDict(CONFIG)
dataset = load_dataset("conll2003")

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

config.vocab_size = len(tokenizer.get_vocab())
config.pad_token_id = tokenizer.get_vocab()['[PAD]']
config.label_list = dataset['train'].features['ner_tags'].feature.names

train_dataset = NERDataset(dataset['train'], tokenizer)
valid_dataset = NERDataset(dataset['validation'], tokenizer)
test_dataset = NERDataset(dataset['test'], tokenizer)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, collate_fn=data_collator)
valid_loader = DataLoader(valid_dataset, config.batch_size, shuffle=True, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, config.batch_size, shuffle=False, collate_fn=data_collator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device is {device}")

model = ModelForNER(config=config)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=config.lr,weight_decay=config.weight_decay)
criterion = CrossEntropyLoss()

model.train_model(train_loader, valid_loader, optimizer, criterion, device, config.num_epochs)
model.validate_model(test_loader, device, is_test=True)