import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.optim.adam import Adam

import random


# ❶ 모델 정의
model = nn.Sequential(nn.Linear(13, 100), nn.ReLU(), nn.Linear(100, 1))
print("model=", model)

# load model from boston.pth
model.load_state_dict(torch.load("boston.pth"))


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


x = pd.DataFrame(
    data,
    columns=[
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ],
)
y = pd.DataFrame(target, columns=["MEDV"])

# print x,y
print("x.shape: ", x.shape)
print("y.shape: ", y.shape)

X = x.iloc[:, :13].values
Y = y["MEDV"].values

# print shape of X, Y
print(f"X.shape:{X.shape} Y.shape:{Y.shape}")

# get random index
index = random.randint(0, len(X) - 1)
prediction = model(torch.FloatTensor(X[index, :13]))
real = Y[index]
# print index
print(f"index:{index}")
print(f"prediction:{prediction.item()} real:{real}")
# print accuracy
print(f"accuracy:{prediction.item()/real*100}%")
