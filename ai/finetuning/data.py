#!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

import os
import torch
from datasets import load_dataset

dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train")
print(len(dataset))
# print initialize 10 rows
for i in range(10):
    print(100*"-")
    print(dataset[i])