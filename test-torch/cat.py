import torch

# Let's assume we have the following tensors:
# z is a noise map of size (batch_size=2, noise_dim=3)
z = torch.tensor([[1, 2, 3], [4, 5, 6]])

# text_embed is a text embedding of size (batch_size=2, embed_dim=2)
text_embed = torch.tensor([[7, 8], [9, 10]])

# We can concatenate them along dimension 1
concatenated = torch.cat((z, text_embed), dim=1)

print(concatenated)
