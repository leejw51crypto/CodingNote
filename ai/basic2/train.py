import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.optim.adam import Adam


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
# x = x[['AGE']]

y = pd.DataFrame(target, columns=["MEDV"])

# print x,y
print("x.shape: ", x.shape)
print("y.shape: ", y.shape)


# ❶ 모델 정의
model = nn.Sequential(nn.Linear(13, 100), nn.ReLU(), nn.Linear(100, 1))
print("model=", model)

# X = dataFrame.iloc[:, :13].values # ❷ 정답을 제외한 특징을 X에 입력
# Y = dataFrame["target"].values    # 데이터프레임의 target의 값을 추출
X = x.iloc[:, :13].values
Y = y["MEDV"].values

batch_size = 100
learning_rate = 0.001

# ❸ 가중치를 수정하기 위한 최적화 정의
optim = Adam(model.parameters(), lr=learning_rate)


# 에포크 반복
for epoch in range(200):
    # 배치 반복
    for i in range(len(X) // batch_size):
        start = i * batch_size  # ➍ 배치 크기에 맞게 인덱스를 지정
        end = start + batch_size

        # 파이토치 실수형 텐서로 변환
        x = torch.FloatTensor(X[start:end])
        y = torch.FloatTensor(Y[start:end])

        optim.zero_grad()  # ❺ 가중치의 기울기를 0으로 초기화
        preds = model(x)  # ❻ 모델의 예측값 계산
        loss = nn.MSELoss()(preds, y)  # ❼ MSE 손실 계산
        loss.backward()
        optim.step()

    if epoch % 20 == 0:
        print(f"epoch{epoch} loss:{loss.item()}")


# save model
torch.save(model.state_dict(), "boston.pth")
print("save ok")
