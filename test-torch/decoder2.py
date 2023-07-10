import torch
from torchviz import make_dot

# Assuming encoder_output and prev_generated_words are tensors with real numbers
encoder_output = torch.tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
prev_generated_words = torch.tensor([[7.0, 8.0],
                                     [9.0, 10.0]])

# Convert tensors to sparse tensors
encoder_output_sparse = encoder_output.to_sparse()
prev_generated_words_sparse = prev_generated_words.to_sparse()

# Concatenating encoder output and previously generated words
combined_input = torch.cat((encoder_output_sparse, prev_generated_words_sparse), dim=1)

# Visualizing the computation graph
make_dot(combined_input).render("decoders", format="png")
