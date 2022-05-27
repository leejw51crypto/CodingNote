import matplotlib.pyplot as plt

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
import logging
import torch
import torch.nn as nn

from torch.optim.adam import Adam

import random


test_data = MNIST(root="./", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)
)
model.to(device)


model.load_state_dict(torch.load("MNIST.pth", map_location=device))


idx = random.randint(0, len(test_data) - 1)


data, label = test_data[idx]
print("data shape=", data.shape, " label=", label)
plt.imshow(data.squeeze(), cmap="gray")
plt.show()
data = torch.reshape(data, (-1, 784)).to(device)
label = torch.tensor([label]).to(device)

with torch.no_grad():
    output = model(data)
    pred = output.argmax(dim=1)
    print("prediction=", pred, " label=", label)
