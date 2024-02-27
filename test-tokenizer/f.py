import torch
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example text
sample_text = "This is a sample text for testing input encoding"

# Tokenize the text
tokenized_text = tokenizer.encode(sample_text, add_special_tokens=False)

# Convert tokenized text to tensor
input_tensor = torch.tensor(tokenized_text).unsqueeze(0)  # Add batch dimension

print("Tokenized text:", tokenized_text)
print("Input tensor shape:", input_tensor.shape)
print("Input tensor:", input_tensor)
