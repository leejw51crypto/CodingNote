import torch
from transformers import CLIPTokenizer, CLIPTextModel

# Load the pre-trained CLIP tokenizer and model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# Example sentence
sentence = "This is a sample sentence."

# Tokenize the sentence
inputs = tokenizer(sentence, return_tensors="pt", padding=True)

# Pass the inputs through the model to get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print("Embeddings Shape:", embeddings.shape)

# Print the final embeddings (first 2 rows, last 2 rows, and ellipsis)
print("Final Embeddings:")
print(embeddings[:, :2, :])
print("...")
print(embeddings[:, -2:, :])
