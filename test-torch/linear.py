import torch
import torch.nn as nn
import torchviz

# Create a linear layer with input size 10 and output size 5
linear_layer = nn.Linear(10, 5)

# Generate some random input data
input_data = torch.randn(32, 10)

# Pass the input data through the linear layer
output = linear_layer(input_data)

# Print the shapes of the tensors
print(f"input_data shape: {input_data.shape}")
print(f"weight shape: {linear_layer.weight.shape}")
print(f"bias shape: {linear_layer.bias.shape}")
print(f"output shape: {output.shape}")

# Visualize computation graph
dummy_input = torch.randn(32, 10)  # Create a dummy input with the same shape as input_data
dot = torchviz.make_dot(output, params=dict(linear_layer.named_parameters()))
dot.format = 'png'  # Specify the image format (e.g., 'png', 'svg')
dot.render(filename='linear')  # Save the computation graph as an image file
