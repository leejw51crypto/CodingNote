import torch
import torch.nn as nn

embed_dim = 256  # Embedding dimension
hidden_dim = 512  # Hidden dimension

# Create a random input tensor
batch_size = 4
sequence_length = 10
input_tensor = torch.randn(batch_size, sequence_length, embed_dim)

# Define the Conv1D layer
conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=2 * hidden_dim, kernel_size=3)

# Apply the Conv1D layer to the input tensor
print(conv1d)
output_tensor = conv1d(input_tensor.transpose(1, 2)).transpose(1, 2)

# Split the output tensor into key and value tensors
key, value = torch.split(output_tensor, hidden_dim, dim=-1)

# Print the shapes of the key and value tensors
print("Key shape:", key.shape)
print("Value shape:", value.shape)

