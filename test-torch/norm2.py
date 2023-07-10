import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features, device="mps"))
        self.beta = nn.Parameter(torch.zeros(features, device="mps"))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta

# Example usage
input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="mps")
layer_norm = LayerNorm(features=3)
output = layer_norm(input_tensor)

print("Input tensor:")
print(input_tensor)
print("\nNormalized output:")
print(output)

