import torch
from datasets import load_dataset
trec = load_dataset("trec", split='train[:1000]')
print(trec)

from transformers import AutoTokenizer, AutoModel
tokenizer= AutoTokenizer.from_pretrained('bert-base-uncased')
model= AutoModel.from_pretrained('bert-base-uncased')

text= trec['text'][:64]
print(text)

tokens= tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors='pt')
print(tokens)


device= torch.device('mps')
model.to(device)
tokens.to(device)
print(device)
model(**tokens)
