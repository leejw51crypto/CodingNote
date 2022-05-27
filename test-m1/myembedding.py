import torch
from torch import nn

device = torch.device("mps")
print(f"device={device}")


# Define vocabulary size and the size of the embedding vectors
vocab_size = 10000
embedding_dim = 128

# Create an embedding layer
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device)

# Example tensor of token indices
# These would usually come from a tokenizer, but here we'll use random values
token_indices = torch.randint(0, vocab_size, (1, 10)).to(device)
print("token_indies.shape=",token_indices.shape)
print("token_indicies=",token_indices)

# Feed the token indices into the embedding layer to get the embeddings
token_embeddings = embedding(token_indices)

print(token_embeddings)
print(token_embeddings.shape)  # Outputs: torch.Size([1, 10, 128])


