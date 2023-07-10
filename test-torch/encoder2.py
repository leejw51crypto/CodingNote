import torch

# Example encoder hidden states
encoder_hidden_states = torch.tensor([
    [[0.1, 0.2, 0.3],   # Hidden state 1
     [0.4, 0.5, 0.6]],

    [[0.7, 0.8, 0.9],   # Hidden state 2
     [1.0, 1.1, 1.2]],

    [[1.3, 1.4, 1.5],   # Hidden state 3
     [1.6, 1.7, 1.8]],

    [[1.9, 2.0, 2.1],   # Hidden state 4
     [2.2, 2.3, 2.4]]
])

# Get the sequence length and hidden dimension from the encoder_hidden_states tensor
sequence_length, hidden_dim = encoder_hidden_states.size(1), encoder_hidden_states.size(2)

# Reshape the encoder hidden states to have a shape of (batch_size, hidden_dim, sequence_length)
encoder_hidden_states = encoder_hidden_states.transpose(1, 2).contiguous()

# Expand the encoder hidden states to match the shape of the decoder inputs
encoder_hidden_states = encoder_hidden_states.unsqueeze(2).expand(-1, -1, sequence_length, -1)

# Print the shape of the encoder hidden states
print("Encoder hidden states shape:", encoder_hidden_states.shape)
