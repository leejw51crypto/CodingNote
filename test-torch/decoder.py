import torch

batch_size = 2
sequence_length = 3
hidden_size=5
generated_length = 4

# Assuming encoder_output and prev_generated_words are tensors
encoder_output = torch.randn(batch_size, sequence_length, hidden_size)
prev_generated_words = torch.randn(batch_size, generated_length, hidden_size)

#print shape of encoder_output and prev_generated_words
print("encoder_output shape: ", encoder_output.shape)
print("prev_generated_words shape: ", prev_generated_words.shape)
# Concatenating encoder output and previously generated words
combined_input = torch.cat((encoder_output, prev_generated_words), dim=1)
print("combined_input shape: ", combined_input.shape)
