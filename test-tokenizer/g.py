import torch
from transformers import GPT2Model, GPT2Tokenizer

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Example text
sample_text = "This is a sample text for testing input encoding"

# Tokenize the text
tokenized_text = tokenizer.encode(sample_text, add_special_tokens=False)
print("Tokenized text:", tokenized_text)

# Convert tokenized text to tensor
input_tensor = torch.tensor(tokenized_text).unsqueeze(0)  # Add batch dimension
print("Input tensor shape:", input_tensor.shape)

# Feed input tensor into the model
output = model(input_tensor)

# Extract the output embeddings
output_embeddings = output.last_hidden_state

print("Output embeddings shape:", output_embeddings.shape)
