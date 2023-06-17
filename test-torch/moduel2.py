import torch
import torch.nn as nn
from torchviz import make_dot

class TwoLayerModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the module
input_size = 3
hidden_size = 5
output_size = 2
module = TwoLayerModule(input_size, hidden_size, output_size)

# Generate some random input data
input_data = torch.randn(1, input_size)

# Visualize the computation graph
output = module(input_data)
dot = make_dot(output, params=dict(module.named_parameters()))
dot.render("module2", format="png")
 
