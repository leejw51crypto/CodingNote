import torch
import torch.nn as nn

# Define the vocabulary
vocab = ['[PAD]', '[UNK]', 'This', 'is', 'a', 'sample', 'sentence']

# Create a mapping from tokens to indices
token_to_idx = {token: idx for idx, token in enumerate(vocab)}

def sentence_to_tokens(sentence):
    tokens = sentence.split()  # Split the sentence into tokens
    return tokens

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, tokens):
        print(tokens)
        indices = [token_to_idx.get(token, token_to_idx['[UNK]']) for token in tokens]
        #print indices
        print(indices)
        indices_tensor = torch.tensor(indices).unsqueeze(0)  # Add batch dimension
        print(indices_tensor)
        embeddings = self.embedding(indices_tensor)
        print(embeddings.shape)
        return embeddings

# Example usage
sentence = "This is a sample sentence"

# Convert sentence to tokens
tokens = sentence_to_tokens(sentence)
print("Tokens:", tokens)

# Create an instance of the EmbeddingLayer
embedding_dim = 128
embedding_layer = EmbeddingLayer(len(vocab), embedding_dim)

# Get the embeddings directly from tokens
embeddings = embedding_layer(tokens)
print("Embeddings Shape:", embeddings.shape)
