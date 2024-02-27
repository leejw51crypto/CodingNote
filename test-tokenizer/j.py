from transformers import BertTokenizer, BertModel
import torch

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "Hello world, this is a test sentence."

# Tokenize the sentence
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

# Print the tokens and token IDs
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("Tokens:", tokens)
print("Token IDs:", inputs["input_ids"])

# Now, let's pass the tokenized input through the BERT model to get the encoder outputs
with torch.no_grad():
    outputs = model(**inputs)

# The encoder outputs are contained in outputs.last_hidden_state
encoder_outputs = outputs.last_hidden_state

# Let's examine the shape of the encoder outputs
print("Shape of Encoder Outputs:", encoder_outputs.shape)
# This will give us [batch_size, seq_length, hidden_size], where hidden_size is 768 for BERT base models

# For a detailed look, let's print the encoder output for the first token ('[CLS]' token)
print("Encoder output for '[CLS]' token shape:", encoder_outputs[0][0].shape)
