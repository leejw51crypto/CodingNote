import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initial prompt
prompt_text = "Once upon a time, there was a"

# Tokenize input prompt
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
print("Input IDs:", input_ids)
# Generate text
max_length = 50
output_ids = model.generate(input_ids, max_length=max_length)

# Decode generated tokens back to text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated text:", generated_text)
