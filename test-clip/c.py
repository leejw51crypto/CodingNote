import torch
from transformers import BertTokenizer, BertModel

# Load a pre-trained tokenizer and model (e.g., BERT)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "This is a sample sentence."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)
print("Tokens:", tokens)

# Convert tokens to indices
indices = tokenizer.convert_tokens_to_ids(tokens)
print("Indices:", indices)

# Add special tokens (e.g., [CLS] and [SEP])
indices = tokenizer.build_inputs_with_special_tokens(indices)
print("Indices with special tokens:", indices)

# Create attention mask
attention_mask = [1] * len(indices)

# Convert indices and attention mask to tensors
indices_tensor = torch.tensor([indices])
attention_mask_tensor = torch.tensor([attention_mask])

# Pass the indices and attention mask through the model to get embeddings
with torch.no_grad():
    outputs = model(indices_tensor, attention_mask=attention_mask_tensor)
    embeddings = outputs.last_hidden_state

print("Embeddings Shape:", embeddings.shape)

# Print the final embeddings
print("Final Embeddings:")
print(embeddings[:, :2, :])
print("...")
print(embeddings[:, -2:, :])
