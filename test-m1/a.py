from datasets import load_dataset
trec = load_dataset("trec", split='train[:1000]')
print(trec)
#features: ['text', 'coarse_label', 'fine_label'],
print(trec[0])