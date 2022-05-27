from datasets import load_dataset
trec = load_dataset("trec", split='train[:1000]')
print(trec)
from transformers import BertForSequenceClassification, BertConfig
config= BertConfig.from_pretrained('bert-base-uncased')
print(trec['coarse_label'])
print(max(trec['coarse_label'])+1 )
config.num_labels= max(trec['coarse_label'])+1 

