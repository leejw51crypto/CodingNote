import torch
import torch.nn as nn
import torch.optim as optim
import torchviz

# Generate random dataset
input_size = 20
output_size = 5
num_samples = 1000

input_data = torch.randn(num_samples, input_size)
target_data = torch.randn(num_samples, output_size)

# Define the LoRa model
class LoRaModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LoRaModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Fully connected layer
        print(f"fc1 input_size, 64 shape:{self.fc1.weight.shape}")
        self.fc2 = nn.Linear(64, output_size)
        print(f"fc2 64, output_size shape: {self.fc1.weight.shape}")

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


print(f"Input data shape: {input_data.shape}")
# Initialize the LoRa model
model = LoRaModel(input_size, output_size)

# Print shape of input_data and target_data
print(f"Target data shape: {target_data.shape}")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate graph visualization
dummy_input = torch.randn(1, input_size)  # Dummy input for graph visualization
torchviz.make_dot(model(dummy_input), params=dict(model.named_parameters())).render("lora", format="png")

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, target_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print training progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
