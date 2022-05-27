import torch
import torch.nn as nn

class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

print("torch has_mps=",torch.has_mps)

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

    # Create a Tensor directly on the mps device
    x = torch.ones((5, 1), device=mps_device)  # 2D tensor (5,1) instead of 1D tensor (5,)
    print(f"x shape={x.shape}")
    print(f"x={x}") 

    # Any operation happens on the GPU
    y = x * 2

    # Move your model to mps just like any other device
    model = SimpleLinearRegression()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)
    print("OK")
