import matplotlib.pyplot as plt

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader

import torch
import torch.nn as nn

from torch.optim.adam import Adam


test_data = MNIST(root="./", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)
)
model.to(device)


model.load_state_dict(torch.load("MNIST.pth", map_location=device))

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        data = torch.reshape(data, (-1, 784)).to(device)

        output = model(data.to(device))
        preds = output.data.max(1)[1]
        print("prediction shape=", preds.shape, " value=", preds)
        print("label shape=", label.shape, " value=", label)
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy:{num_corr/len(test_data)*100}%")  # 분류 정확도를 출력합니다.
