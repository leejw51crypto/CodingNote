import torch
import torch.nn as nn
from torchviz import make_dot

class OneLayerModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(OneLayerModule, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

# Create an instance of the module
input_size = 3
output_size = 2
module = OneLayerModule(input_size, output_size)

# Generate some random input data
input_data = torch.randn(1, input_size)

# Visualize the computation graph
output = module(input_data)
dot = make_dot(output, params=dict(module.named_parameters()))
dot.render("one", format="png")

