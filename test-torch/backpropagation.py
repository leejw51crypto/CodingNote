import numpy as np
import torch
import torchviz
from torch import nn


class AffineLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(AffineLayer, self).__init__()  # Call the constructor of nn.Module
        self.input_size = input_size
        self.output_size = output_size
        self.weights = nn.Parameter(torch.randn(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, inputs):
        return torch.matmul(inputs, self.weights) + self.bias


# Example usage
# Create an instance of the AffineLayer with input size of 2 and output size of 3
layer = AffineLayer(2, 3)

# Forward pass
x = torch.tensor([[1, 2]], dtype=torch.float32)
output = layer.forward(x)
print("Forward pass output:")
print(output)

# Backward pass
grad_output = torch.ones_like(output)
learning_rate = 0.1
output.backward(gradient=grad_output)
grad_input = x.grad
print("Backward pass gradient:")
print(grad_input)

# Updated weights and bias
print("Updated weights:")
print(layer.weights)
print("Updated bias:")
print(layer.bias)


# Visualize computation graph
dummy_input = torch.randn(1, 2)  # Create a dummy input
dot = torchviz.make_dot(layer(dummy_input), params=dict(layer.named_parameters()))
dot.format = 'png'  # Specify the image format (e.g., 'png', 'svg')
dot.render(filename='backpropagation')  # Save the computation graph as an image file
