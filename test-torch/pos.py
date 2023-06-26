import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchviz import make_dot

# Define the positional encoding function
def positional_encoding(seq_length, d_model):
    encoding = torch.zeros(seq_length, d_model)
    pos = torch.arange(seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
    encoding[:, 0::2] = torch.sin(pos * div_term)
    encoding[:, 1::2] = torch.cos(pos * div_term)
    return encoding

# Define the text to be encoded
text = "Hello, how are you doing today?"
print("Text:", text)

# Tokenize the text
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)

# Create token-to-index mapping
token_to_idx = {token: idx for idx, token in enumerate(set(tokens))}

# Convert tokens to token IDs
token_ids = torch.tensor([token_to_idx[token] for token in tokens])

print("Token IDs Shape:", token_ids.shape)
print("Token IDs:", token_ids)

# Convert token IDs to token embeddings
embedding_dim = 16
embedding = nn.Embedding(len(token_to_idx), embedding_dim)
token_embeddings = embedding(token_ids)

print("Token Embeddings Shape:", token_embeddings.shape)
print("Token Embeddings:", token_embeddings)

# Apply positional encoding
seq_length = len(token_ids)
d_model = embedding_dim
pos_encoding = positional_encoding(seq_length, d_model)

print("Positional Encoding Shape:", pos_encoding.shape)
print("Positional Encoding:", pos_encoding)

# Add positional encoding to the token embeddings
token_embeddings = token_embeddings + pos_encoding

print("Token Embeddings with Positional Encoding Shape:", token_embeddings.shape)
print("Token Embeddings with Positional Encoding:", token_embeddings)

# Visualize the computational graph
dot = make_dot(token_embeddings, params=dict(embedding.named_parameters()))
dot.render(filename='positional_encoding_graph', format='png')

print("Final Token Embeddings Shape:", token_embeddings.shape)
print("Final Token Embeddings:", token_embeddings)
