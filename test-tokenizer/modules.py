import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the first complex module
class MyComplexNet(nn.Module):
    def __init__(self):
        super(MyComplexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1024, 128)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout1(x)
        x = self.fc1(x)
        return F.relu(x)

# Define the second complex module
class MyComplexNet2(nn.Module):
    def __init__(self):
        super(MyComplexNet2, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Define the overall network that connects MyComplexNet and MyComplexNet2
class MyFullNetwork(nn.Module):
    def __init__(self):
        super(MyFullNetwork, self).__init__()
        self.block1 = MyComplexNet()
        self.block2 = MyComplexNet2()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# Instantiate the network
model = MyFullNetwork()

# Example input: a batch of 10 images, 1 channel, 28x28 pixels
example_input = torch.randn(10, 1, 28, 28)

# Forward pass
output = model(example_input)

print("Output shape:", output.shape)
