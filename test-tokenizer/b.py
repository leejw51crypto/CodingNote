import torch
from transformers import BertModel, BertTokenizer

# Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample text
text = "hello world !"

# Tokenize the text and convert to tensor
encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=True)

# Get the embeddings for each token
with torch.no_grad():
    outputs = model(**encoded_input)

# The last hidden state is the embeddings of the tokens in the sequence
embeddings = outputs.last_hidden_state
print(embeddings.shape)  # Shape: (batch_size, sequence_length, hidden_size)

# Accessing the embedding for the first token ([CLS] token in this case)
cls_embedding = embeddings[:, 0, :]
print(cls_embedding)
