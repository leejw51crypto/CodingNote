import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Input sentence
input_sentence = "This is an example sentence."

# Step 1: Tokenization
tokens = tokenizer.tokenize(input_sentence)

# Step 2: Adding Special Tokens (if required)
tokens = ['[CLS]'] + tokens + ['[SEP]']

# Step 3: Padding (if required)
max_length = 512  # Maximum sequence length supported by BERT
padded_tokens = tokens + ['[PAD]'] * (max_length - len(tokens))

# Step 4: Conversion to IDs
input_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
print(f"Input IDs: {input_ids}")
print(f"Input IDs length: {len(input_ids)}")

# Step 5: Embedding Lookup
input_ids_tensor = torch.tensor([input_ids])  # Convert to tensor if using PyTorch
with torch.no_grad():
    outputs = model(input_ids_tensor)

print(f"Output shape: {outputs[0].shape}")
# Extract the embeddings for the [CLS] token which represents the entire sequence
embeddings = outputs[0][:, 0, :]  # Extract embeddings for [CLS] token


#print(embeddings)
print(f"Embeddings shape: {embeddings.shape}")
