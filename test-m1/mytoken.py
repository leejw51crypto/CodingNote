from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("mps")
print(f"device={device}")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model = model.to(device)
print(f"model={model}")

short_sentence = "soccer game"
long_sentence = "This is a really long sentence that is used to demonstrate the effect of the max_length parameter in the BERT tokenizer. It goes on and on, far beyond what would be considered a typical sentence length. You can see that it is significantly longer than the first sentence."

short_tokens = tokenizer(short_sentence, max_length=20, padding='max_length', truncation=True)
long_tokens = tokenizer(long_sentence, max_length=10, padding='max_length', truncation=True)

print(short_tokens)
print(long_tokens)

# Convert short tokens to embeddings
short_input_ids = torch.tensor(short_tokens['input_ids']).unsqueeze(0).to(device)  # Add batch dimension
short_outputs = model(short_input_ids)
short_embeddings = short_outputs.last_hidden_state
print("Short sentence embeddings shape:", short_embeddings.shape)

# Convert long tokens to embeddings
long_input_ids = torch.tensor(long_tokens['input_ids']).unsqueeze(0).to(device)  # Add batch dimension
long_outputs = model(long_input_ids)
long_embeddings = long_outputs.last_hidden_state
print("Long sentence embeddings shape:", long_embeddings.shape)
