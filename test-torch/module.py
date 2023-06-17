import torch
import torch.nn as nn

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

# Define the loss criterion
criterion = nn.MSELoss()

# Generate some random input data and target
input_data = torch.randn(1, input_size)
target = torch.randn(1, output_size)

# Forward pass
output = module(input_data)

# Compute the loss
loss = criterion(output, target)

# Backward pass (compute gradients)
module.zero_grad()  # Clear existing gradients
loss.backward()

# Print the gradients
print(module.fc1.weight.grad)
print(module.fc1.bias.grad)
print(module.fc2.weight.grad)
print(module.fc2.bias.grad)
