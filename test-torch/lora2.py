import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pretrained GPT-2 model and tokenizer
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Example text prompt for adaptation
prompt = "Once upon a time"

# Encode the prompt into input tensors
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate initial output using the pretrained model
with torch.no_grad():
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Extract the generated text from the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Perform low-rank adaptation
rank = 16  # desired rank for low-rank adaptation
model.transformer.set_output_dim(rank)

# Fine-tune the model on the generated text
input_ids_finetune = tokenizer.encode(generated_text, return_tensors="pt").to(device)
labels = input_ids_finetune.clone()
labels[:, :-1] = -100  # Ignore loss for input tokens
labels = labels.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()
for _ in range(10):  # Perform 10 iterations of fine-tuning
    optimizer.zero_grad()
    outputs = model(input_ids_finetune, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Generate text using the adapted model
with torch.no_grad():
    output_adapted = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Extract the adapted generated text
generated_text_adapted = tokenizer.decode(output_adapted[0], skip_special_tokens=True)

# Print the results
print("Generated text before adaptation:")
print(generated_text)
print()
print("Generated text after adaptation:")
print(generated_text_adapted)
