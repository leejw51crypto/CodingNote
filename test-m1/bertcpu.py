import torch
from transformers import AutoTokenizer, AutoModel

#device = torch.device('mps')
device = torch.device('cpu')
print("device=",device)

from datasets import load_dataset
trec = load_dataset("trec", split='train[:1000]')
print(trec)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer(
    trec['text'], max_length=512,
    truncation=True, padding='max_length'
)

import numpy as np
labels = np.zeros(
    (len(trec), max(trec['coarse_label'])+1)
)
# one-hot encode
labels[np.arange(len(trec)), trec['coarse_label']] = 1


class TrecDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, idx):
        input_ids = self.tokens[idx].ids
        attention_mask = self.tokens[idx].attention_mask
        labels = self.labels[idx]
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels, dtype=torch.float32)  # specify datatype here
        
        }

    def __len__(self):
        return len(self.labels)

dataset = TrecDataset(tokens, labels)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=64
)
from transformers import BertForSequenceClassification, BertConfig

config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = max(trec['coarse_label'])+1
model = BertForSequenceClassification(config).to(device)  # remember to move to MPS!

for param in model.bert.parameters():
    param.requires_grad = False

# activate training mode of model
model.train()

# initialize adam optimizer with weight decay
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

from time import time
from tqdm.auto import tqdm

loop_time = []

# setup loop (using tqdm for the progress bar)
loop = tqdm(loader, leave=True)
for batch in loop:
    batch_mps = {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].to(device)
    }
    t0 = time()
    # initialize calculated gradients (from prev step)
    optim.zero_grad()
    # train model on batch and return outputs (incl. loss)
    outputs = model(**batch_mps)
    # extract loss
    loss = outputs[0]
    # calculate loss for every parameter that needs grad update
    loss.backward()
    # update parameters
    optim.step()
    loop_time.append(time()-t0)
    # print relevant info to progress bar
    loop.set_postfix(loss=loss.item())    