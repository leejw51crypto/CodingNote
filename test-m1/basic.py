# Import PyTorch
import torch

# Check if a CUDA-compatible GPU is available, and if not, use the CPU
#device = torch.device("cuda" if torch.mps.is_available() else "cpu")
device = torch.device("mps")

# Create a tensor and move it to the chosen device
x = torch.tensor([1, 2, 3]).to(device)
print(f"x is on {device}")

# Perform operations on the tensor
y = x ** 2
print(y)

